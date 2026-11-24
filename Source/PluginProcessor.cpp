#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <algorithm>
#include <cmath>

DeReverbenjaminV2Processor::DeReverbenjaminV2Processor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMETERS", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout DeReverbenjaminV2Processor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Selector de Fuente (0: DRUMS, 1: VOCAL, 2: GTR/BASS, 3: KEYS)
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{"mode", 1}, "Source Mode",
        juce::StringArray{"DRUMS", "VOCAL", "GTR/BASS", "KEYS"}, 0));

    // Etapa 1: Spectral De-Room
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"roomAmount", 1}, "De-Room",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.55f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"focus", 1}, "Ambiance",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.45f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"decay", 1}, "Room Decay",
        juce::NormalisableRange<float>(0.05f, 0.95f, 0.01f), 0.50f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"smooth", 1}, "Smooth",
        juce::NormalisableRange<float>(0.0f, 0.95f, 0.01f), 0.55f));

    // Etapa 2: Tail De-Reverb & Output
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"tailAmount", 1}, "De-Tail",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.30f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"sensitivity", 1}, "Sensitivity",
        juce::NormalisableRange<float>(0.1f, 2.0f, 0.01f), 1.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"release", 1}, "Release",
        juce::NormalisableRange<float>(20.0f, 400.0f, 1.0f), 110.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"output", 1}, "Output",
        juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f), 0.0f));

    // Etapa 3: Reconstruct & Analog Restore
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"punch", 1}, "Punch",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.35f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"warmth", 1}, "Warmth",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.30f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"air", 1}, "Air",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.25f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"mix", 1}, "Mix",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 1.0f));

    return { params.begin(), params.end() };
}

void DeReverbenjaminV2Processor::prepareToPlay(double sampleRate, int)
{
    currentSampleRate = sampleRate;
    setLatencySamples(fftSize);

    channels.resize(static_cast<size_t>(std::max(1, getTotalNumInputChannels())));
    for (auto& ch : channels)
        ch.reset();

    fastEnv = 0.0f;
    slowEnv = 0.0f;
    tailSmoothGain = 1.0f;
    dryTransFast = 0.0f;
    dryTransSlow = 0.0f;
    currentGainReduction.store(1.0f);
}

bool DeReverbenjaminV2Processor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono()
        || layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void DeReverbenjaminV2Processor::processFrame(ChannelState& state, float amount, float roomDecay,
                                              float roomFocus, float smoothness, int mode)
{
    std::copy(state.inputBuffer.begin(), state.inputBuffer.end(), state.fftData.begin());
    std::fill(state.fftData.begin() + fftSize, state.fftData.end(), 0.0f);
    window.multiplyWithWindowingTable(state.fftData.data(), fftSize);

    fft.performRealOnlyForwardTransform(state.fftData.data(), true);

    // Calibración espectral según el instrumento seleccionado
    const float binHz = static_cast<float>(currentSampleRate) / static_cast<float>(fftSize);
    float modeFloor = 0.08f; // Piso mínimo anti-artefactos
    if (mode == 1) modeFloor = 0.12f;      // VOCAL: piso más suave para evitar sonido robótico
    else if (mode == 2) modeFloor = 0.10f; // GTR/BASS: protege sustain
    else if (mode == 3) modeFloor = 0.14f; // KEYS: evita cortes bruscos en acordes

    const float overSub = 0.55f + roomFocus * 2.2f;
    const float minGain = std::max(modeFloor, 1.0f - (amount * 0.92f));
    const float timeSmooth = 0.28f + smoothness * 0.62f;

    for (int k = 0; k < numBins; ++k)
    {
        const float freq = k * binHz;
        const float re = state.fftData[2 * k];
        const float im = state.fftData[2 * k + 1];
        const float power = re * re + im * im;

        // Protección por banda según instrumento
        float bandWeight = 1.0f;
        if (mode == 0 && freq < 110.0f)                          bandWeight = 0.45f; // DRUMS: protege sub-bombo
        else if (mode == 1 && freq >= 160.0f && freq <= 3500.0f) bandWeight = 0.78f; // VOCAL: protege cuerpo/inteligibilidad
        else if (mode == 2 && freq >= 100.0f && freq <= 2200.0f) bandWeight = 0.72f; // GTR/BASS: protege notas y cuerpo
        else if (mode == 3 && freq < 250.0f)                     bandWeight = 0.65f; // KEYS: protege graves del piano/synth

        state.reverbEst[k] = std::max(roomDecay * state.reverbEst[k],
                                      0.35f * state.reverbEst[k] + 0.65f * state.prevPower[k]);
        state.prevPower[k] = power;

        const float estRoom = state.reverbEst[k] * overSub * bandWeight;
        const float cleanPower = std::max(0.0f, power - estRoom);
        const float wiener = cleanPower / (power + 1e-9f);

        state.rawGain[k] = juce::jlimit(minGain, 1.0f, (1.0f - amount) + amount * wiener);
    }

    // Suavizado espectral de 5 puntos (reduce mucho el "musical noise" en agudos)
    float gainSum = 0.0f;
    for (int k = 0; k < numBins; ++k)
    {
        float g = state.rawGain[k];
        if (k > 1 && k < numBins - 2)
        {
            g = 0.10f * state.rawGain[k - 2] + 0.20f * state.rawGain[k - 1]
              + 0.40f * state.rawGain[k]
              + 0.20f * state.rawGain[k + 1] + 0.10f * state.rawGain[k + 2];
        }

        state.smoothGain[k] = timeSmooth * state.smoothGain[k] + (1.0f - timeSmooth) * g;
        gainSum += state.smoothGain[k];

        state.fftData[2 * k]     *= state.smoothGain[k];
        state.fftData[2 * k + 1] *= state.smoothGain[k];
    }
    state.lastAvgGain = gainSum / static_cast<float>(numBins);

    fft.performRealOnlyInverseTransform(state.fftData.data());
    window.multiplyWithWindowingTable(state.fftData.data(), fftSize);

    constexpr float normFactor = 2.0f / 3.0f;
    for (int i = 0; i < fftSize; ++i)
        state.outputAccum[i] += state.fftData[i] * normFactor;

    std::copy(state.outputAccum.begin(), state.outputAccum.begin() + hopSize, state.outputReady.begin());
    std::copy(state.outputAccum.begin() + hopSize, state.outputAccum.end(), state.outputAccum.begin());
    std::fill(state.outputAccum.end() - hopSize, state.outputAccum.end(), 0.0f);
    std::copy(state.inputBuffer.begin() + hopSize, state.inputBuffer.end(), state.inputBuffer.begin());
}

void DeReverbenjaminV2Processor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int mode = static_cast<int>(apvts.getRawParameterValue("mode")->load());
    const float roomAmount = apvts.getRawParameterValue("roomAmount")->load();
    const float focus = apvts.getRawParameterValue("focus")->load();
    const float decay = apvts.getRawParameterValue("decay")->load();
    const float smooth = apvts.getRawParameterValue("smooth")->load();

    const float tailAmount = apvts.getRawParameterValue("tailAmount")->load();
    const float sensitivity = apvts.getRawParameterValue("sensitivity")->load();
    const float releaseMs = apvts.getRawParameterValue("release")->load();
    const float outGain = juce::Decibels::decibelsToGain(apvts.getRawParameterValue("output")->load());

    const float punch = apvts.getRawParameterValue("punch")->load();
    const float warmth = apvts.getRawParameterValue("warmth")->load();
    const float air = apvts.getRawParameterValue("air")->load();
    const float mix = apvts.getRawParameterValue("mix")->load();

    const int numChannels = std::min(buffer.getNumChannels(), static_cast<int>(channels.size()));
    const int numSamples = buffer.getNumSamples();

    // Constantes de tiempo adaptadas al instrumento
    const float sr = static_cast<float>(currentSampleRate);
    const float fastAttack = std::exp(-1.0f / (sr * (mode == 0 ? 0.002f : 0.004f)));
    const float fastRelease = std::exp(-1.0f / (sr * 0.040f));
    const float slowAttack = std::exp(-1.0f / (sr * 0.050f));
    const float slowRelease = std::exp(-1.0f / (sr * (releaseMs * 0.001f)));
    const float gainSmoothCoeff = std::exp(-1.0f / (sr * 0.015f));

    // Detector de transientes de Etapa 3 (más rápido en batería, más suave en voz/teclado)
    const float transFastRel = std::exp(-1.0f / (sr * (mode == 0 ? 0.012f : 0.022f)));
    const float transSlowRel = std::exp(-1.0f / (sr * (mode == 0 ? 0.065f : 0.110f)));

    // Frecuencia de corte del excitador de Aire según instrumento
    const float airCutoff = (mode == 0) ? 5500.0f : (mode == 1 ? 7500.0f : (mode == 2 ? 4200.0f : 6500.0f));
    const float airLpCoeff = std::exp(-2.0f * juce::MathConstants<float>::pi * airCutoff / sr);

    float spectralGainAvg = 1.0f;

    for (int i = 0; i < numSamples; ++i)
    {
        float dryPeak = 0.0f;
        float wetPeak = 0.0f;
        float dryAlignedSamples[2] = { 0.0f, 0.0f };
        float wetSamples[2] = { 0.0f, 0.0f };

        // --- ETAPA 1: Spectral De-Room + Señal Seca Alineada ---
        float frameGainSum = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto& state = channels[static_cast<size_t>(ch)];
            const float inSample = buffer.getSample(ch, i);

            // Guardar y leer señal seca con retardo exacto de fftSize muestras
            const float dryAligned = state.dryDelay[static_cast<size_t>(state.dryWritePos)];
            state.dryDelay[static_cast<size_t>(state.dryWritePos)] = inSample;
            state.dryWritePos = (state.dryWritePos + 1) & (fftSize - 1);

            state.inputBuffer[fftSize - hopSize + state.hopIndex] = inSample;
            const float spectralOut = state.outputReady[state.hopIndex];
            state.hopIndex++;

            if (state.hopIndex == hopSize)
            {
                state.hopIndex = 0;
                processFrame(state, roomAmount, decay, focus, smooth, mode);
            }

            if (ch < 2)
            {
                dryAlignedSamples[ch] = dryAligned;
                wetSamples[ch] = spectralOut;
            }
            dryPeak = std::max(dryPeak, std::abs(dryAligned));
            wetPeak = std::max(wetPeak, std::abs(spectralOut));
            frameGainSum += state.lastAvgGain;
        }
        if (numChannels > 0)
            spectralGainAvg = frameGainSum / static_cast<float>(numChannels);

        // --- ETAPA 2: Dual-Envelope Tail De-Reverb ---
        fastEnv = (wetPeak > fastEnv)
            ? fastAttack * fastEnv + (1.0f - fastAttack) * wetPeak
            : fastRelease * fastEnv + (1.0f - fastRelease) * wetPeak;

        slowEnv = (wetPeak > slowEnv)
            ? slowAttack * slowEnv + (1.0f - slowAttack) * wetPeak
            : slowRelease * slowEnv + (1.0f - slowRelease) * wetPeak;

        float ratio = fastEnv / (slowEnv + 1e-6f);
        float targetGain = 1.0f;
        if (ratio < sensitivity)
        {
            float reduction = juce::jlimit(0.0f, 1.0f, ratio / sensitivity);
            targetGain = (1.0f - tailAmount) + (tailAmount * reduction * reduction);
        }
        tailSmoothGain = gainSmoothCoeff * tailSmoothGain + (1.0f - gainSmoothCoeff) * targetGain;

        // --- ETAPA 3: Reconstruct & Analog Restore ---
        // 3A. Detección de transiente sobre la señal seca original
        dryTransFast = (dryPeak > dryTransFast) ? dryPeak : transFastRel * dryTransFast + (1.0f - transFastRel) * dryPeak;
        dryTransSlow = (dryPeak > dryTransSlow) ? (0.85f * dryTransSlow + 0.15f * dryPeak) : transSlowRel * dryTransSlow + (1.0f - transSlowRel) * dryPeak;
        const float transientAmount = juce::jlimit(0.0f, 1.0f, (dryTransFast - dryTransSlow) * 4.5f);

        // Cuánta energía recortó la FFT (sirve para rellenar armónicos solo donde hizo falta)
        const float spectralLoss = juce::jlimit(0.0f, 1.0f, 1.0f - spectralGainAvg);
        const float dynamicDrive = 1.0f + warmth * (1.8f + spectralLoss * 2.2f);

        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto& state = channels[static_cast<size_t>(ch)];
            const float dry = (ch < 2) ? dryAlignedSamples[ch] : 0.0f;
            float sig = ((ch < 2) ? wetSamples[ch] : 0.0f) * tailSmoothGain;

            // Rescate de transientes (Punch): inyecta el ataque seco original sin la cola
            sig += (dry - sig) * (transientAmount * punch * 0.85f);

            // Relleno armónico (Warmth): saturación analógica suave que enmascara huecos espectrales
            if (warmth > 0.001f)
            {
                const float driven = sig * dynamicDrive;
                const float sat = std::tanh(driven + 0.12f * driven * driven) / std::tanh(dynamicDrive);
                sig = sig * (1.0f - warmth * 0.65f) + sat * (warmth * 0.65f);
            }

            // Restauración de brillo seco (Air): excitador de agudos sobre la señal ya limpia
            state.airLpState = airLpCoeff * state.airLpState + (1.0f - airLpCoeff) * sig;
            const float highBand = sig - state.airLpState;
            if (air > 0.001f)
            {
                const float excitedAir = std::tanh(highBand * 2.2f) * 0.55f;
                sig += excitedAir * air * (0.65f + spectralLoss * 0.5f);
            }

            // Mezcla paralela en fase perfecta + Ganancia de salida
            const float finalOut = (dry * (1.0f - mix) + sig * mix) * outGain;
            buffer.setSample(ch, i, finalOut);
        }
    }

    currentGainReduction.store(juce::jlimit(0.0f, 1.0f, spectralGainAvg * tailSmoothGain));
}

juce::AudioProcessorEditor* DeReverbenjaminV2Processor::createEditor()
{
    return new DeReverbenjaminV2Editor(*this);
}

void DeReverbenjaminV2Processor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void DeReverbenjaminV2Processor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DeReverbenjaminV2Processor();
}
