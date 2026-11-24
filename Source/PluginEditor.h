#pragma once
#include "PluginProcessor.h"
#include <juce_gui_basics/juce_gui_basics.h>

class AnalogKnobLAF : public juce::LookAndFeel_V4
{
public:
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, const float rotaryStartAngle,
                          const float rotaryEndAngle, juce::Slider& slider) override;
};

class DeReverbenjaminV2Editor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit DeReverbenjaminV2Editor(DeReverbenjaminV2Processor&);
    ~DeReverbenjaminV2Editor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void setupKnob(juce::Slider& slider, juce::Label& label, const juce::String& name, const juce::String& suffix = "");
    void drawScrew(juce::Graphics& g, float cx, float cy) const;
    void drawVuMeter(juce::Graphics& g, juce::Rectangle<float> area) const;

    DeReverbenjaminV2Processor& audioProcessor;
    AnalogKnobLAF analogLAF;

    // Selector de Fuente
    juce::Slider modeSlider;
    juce::Label modeLabel;

    // Etapa 1, 2 y 3
    juce::Slider roomAmountSlider, focusSlider, decaySlider, smoothSlider;
    juce::Slider tailAmountSlider, sensSlider, releaseSlider, outputSlider;
    juce::Slider punchSlider, warmthSlider, airSlider, mixSlider;

    juce::Label roomAmountLabel, focusLabel, decayLabel, smoothLabel;
    juce::Label tailAmountLabel, sensLabel, releaseLabel, outputLabel;
    juce::Label punchLabel, warmthLabel, airLabel, mixLabel;

    using Attachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    std::unique_ptr<Attachment> modeAtt;
    std::unique_ptr<Attachment> roomAmountAtt, focusAtt, decayAtt, smoothAtt;
    std::unique_ptr<Attachment> tailAmountAtt, sensAtt, releaseAtt, outputAtt;
    std::unique_ptr<Attachment> punchAtt, warmthAtt, airAtt, mixAtt;

    float smoothedGr = 1.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeReverbenjaminV2Editor)
};
