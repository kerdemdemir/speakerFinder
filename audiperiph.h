#ifndef AUDIPERIPH_H
#define AUDIPERIPH_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <fftw3.h>

#define MINCF			80
#define MAXCF			8000


double HzToERBRate(double Hz);
double ERBRateToHz(double ERBRate);

class SignalBank {

public:

    bool init(int channel_count, int signal_length, float sample_rate)
    {
        sample_rate_ = sample_rate;
        buffer_length_ = signal_length;
        channel_count_ = channel_count;
        signals_.resize(channel_count_);
        centre_frequencies_.resize(channel_count_, 0.0f);
        for (int i = 0; i < channel_count_; ++i)
        {
          signals_[i].resize(buffer_length_, 0.0f);
        }
        initialized_ = true;
        return true;
    }

    float sample(int channel, int index) const {
      return signals_[channel][index];
    }

    void set_sample(int channel, int index, float value) {
      signals_[channel][index] = value;
    }

    float sample_rate() const {
      return sample_rate_;
    }

    int buffer_length() const {
      return buffer_length_;
    }

    float centre_frequency(int i) const {
      if (i < channel_count_)
        return centre_frequencies_[i];
      else
        return 0.0f;
    }

    void set_centre_frequency(int i, float cf) {
      if (i < channel_count_)
        centre_frequencies_[i] = cf;
    }

    bool initialized() const {
      return initialized_;
    }

    int channel_count() const {
      return channel_count_;
    }

    void calc_power()
    {
        powers.resize(channel_count_);
        for (int k = 0; k < channel_count_; k++)
        {
            powers[k] = 0;
            for (int j = 0; j < buffer_length_; j++)
            {
                 powers[k] += signals_[k][j] * signals_[k][j];
            }
            //perfectiveLoudless(k);
            powers[k] = log10(powers[k]);
        }
    }

    void perfectiveLoudless(int channel)
    {
        double omega = centre_frequencies_[channel] * 3.147 * 2;
        double million = 1000000;
        double numerator = (pow(omega, 2.0) + 56.8 * million)* pow(omega,4.0);
        double denomirator = pow((pow(omega, 2) + 6.3* million),2.0) * (pow(omega,2.0) + 38*million);
        double E = numerator / denomirator;
        powers[channel] = E * powers[channel];
    }

    double get_power(int channel)
    {
        return powers[channel];
    }

    std::vector<double>& getpowerAll()
    {
        return powers;
    }

private:

    int channel_count_;
    int buffer_length_;
    std::vector<double> centre_frequencies_;
    std::vector<double> powers;
    float sample_rate_;
    bool initialized_;
    std::vector<std::vector<double> > signals_;

};




class Gammatone
{
public:

    bool init(int numberOfChannels, float sampleRate)
    {
        num_channels_ = numberOfChannels;
        max_frequency_ = MAXCF;
        min_frequency_ = MINCF;
        window_size_ = 20 * sampleRate / 1000;;
        sample_rate_ = sampleRate;
        return initiliaze();
    }
    void startProcess(std::vector<double>::const_iterator inputData, size_t sizeOfData, std::vector<double>& output);

private:

    void process(std::vector<double>::const_iterator inputData, SignalBank& output);
    void reverse();
    bool initiliaze();

    // Filter coefficients
    std::vector<std::vector<double> > b1_;
    std::vector<std::vector<double> > b2_;
    std::vector<std::vector<double> > b3_;
    std::vector<std::vector<double> > b4_;
    std::vector<std::vector<double> > a_;

    std::vector<std::vector<double> > state_1_;
    std::vector<std::vector<double> > state_2_;
    std::vector<std::vector<double> > state_3_;
    std::vector<std::vector<double> > state_4_;

    std::vector<double> centre_frequencies_;
    float sample_rate_;
    int num_channels_;
    int window_size_;
    int window_count_;
    double max_frequency_;
    double min_frequency_;
};

#endif // AUDIPERIPH_H

