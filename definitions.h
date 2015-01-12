#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <aubio.h>
#include <sndfile.h>
#include <string>
#include <iostream>
#include <fstream>
#include <mlpack/methods/gmm/gmm.hpp>
#include <audiperiph.h>
#include <fftw3.h>

using namespace mlpack;
using namespace mlpack::gmm;

#define JUMPSIZE 10
#define STATE_COUNT ((450 - 80) / JUMPSIZE)
#define LOAD 1
#define NUMBER_OF_MFCC_FEATURES 32
#define NUMBER_OF_PEOPLE 6

enum class Method
{
    PITCH = 0,
    MFCC = 1,
    PGRAMS = 2,
    ALL_TRAINER = 3
};


inline
const char* method2Str(Method tempMethod)
{
    if(tempMethod == Method::PITCH)
        return "PITCH";
    else if (tempMethod == Method::MFCC)
        return "MFCC";
    else if (tempMethod == Method::PGRAMS)
        return "PGRAMS";
    else
        return "LEARNER";
}



struct featureOutput
{
    arma::mat data;
    std::vector<double> pitchs;
    std::vector<size_t> pitchNormalized;
    arma::mat pitchData;
    arma::mat resultData;
    size_t lastPosResultData;
    std::vector<double> rawData;
    double *fftwData;
    fftw_plan fftwPlan;


    int
    openWavFile(std::string fileName, int& sampleRate)
    {
        double buffer[100000];
        SF_INFO soundFileInfo;
        SNDFILE* soundFile ;


        soundFileInfo.format = 0;
        soundFile = sf_open(fileName.c_str(), SFM_READ , &soundFileInfo);


        if ( sf_error (soundFile)  != 0)
        {
            std::cout << "Problem with input wav file" << std::endl;
            std::cout << "Error:" << sf_error_number(sf_error (soundFile) ) << std::endl;
            return -1;
        }

        printf ("    Sample rate : %d\n", soundFileInfo.samplerate) ;
        sampleRate = soundFileInfo.samplerate;
        int readCount = sf_read_double(soundFile, buffer, 100000);
        if (readCount <= 0)
        {
            std::cout << "Problem while reading the file" << std::endl;
            return -1;
        }

        rawData.assign(buffer,&buffer[readCount]);
        rawData.resize(readCount);
        return readCount;

    }

    void getAllInOne()
    {
        lastPosResultData = 0;
        size_t smallestSize = data.n_cols;

        if (smallestSize > pitchData.n_cols)
            smallestSize = pitchData.n_cols;

        if (smallestSize > pitchNormalized.size())
            smallestSize = pitchNormalized.size();

        resultData.resize(3,smallestSize);
    }

    void
    pushResultData(double probibilty, double probAll, Method method)
    {
        if (lastPosResultData >= resultData.n_cols)
            return;


        size_t methodInt = (size_t)method;
        if (probAll <= 0)
            resultData(methodInt, lastPosResultData) = 0;
        else
            resultData(methodInt, lastPosResultData) = probibilty;

        if (methodInt >= (resultData.n_rows - 1))
            lastPosResultData++;
    }

    void calcPitchData()
    {
        pitchs.erase(std::remove_if(pitchs.begin(), pitchs.end(), [](double i) {return !(i >= 80 && i <= 450);}),
                    pitchs.end());
        pitchData.resize(1, pitchs.size());
        for (size_t i = 0; i < pitchs.size(); i++)
        {
            pitchNormalized.push_back((pitchs[i] - 80) / JUMPSIZE);
            pitchData(0, i) = pitchs[i];
        }
        getAllInOne();
    }

    void dctStep(size_t counter, std::vector<double>& input)
    {
        fftwPlan = fftw_plan_r2r_1d(input.size(), input.data(), fftwData, FFTW_REDFT10, FFTW_MEASURE);
        fftw_execute(fftwPlan);

        for (int i = 0; i < NUMBER_OF_MFCC_FEATURES; i++)
        {
           data(i, counter) = fftwData[i];
        }
        fftw_destroy_plan(fftwPlan);
    }

    int getFeatures(std::string& fileName, Gammatone& gammatone)
    {
        int samplerate = 16000; // samplerate
        if (openWavFile(fileName, samplerate) == -1)
        {
            std::cout << "Problem while opening the sound file " << std::endl;
            return -1;
        }



        //uint_t hop_s = win_s / 4; // hop size
        uint_t hop_s = 256;
        uint_t win_s = hop_s * 4; // window size
        // create some vectors
        fvec_t *input = new_fvec (hop_s); // input buffer
        fvec_t *out = new_fvec (1); // output candidates
        // create pitch object
        aubio_pitch_t *o = new_aubio_pitch ("default", win_s, hop_s, samplerate);
        // 2. do something with it
        fftwData = (double *)fftw_malloc(sizeof (double) * hop_s);
        int counter = 0;

        data.resize(NUMBER_OF_MFCC_FEATURES, (rawData.size() / hop_s + 1));
        std::vector<double> gammaCoef;
        std::vector<double>::iterator itePos = rawData.begin();
        for (size_t i = 0; i < rawData.size(); i += hop_s)
        {

            for (size_t k = 0; k < hop_s; k++)
            {
                size_t currentPos =  i + k;
                if (currentPos > rawData.size())
                {
                    break;
                }
                *(input->data + k) = *(rawData.begin() + currentPos);
            }

            aubio_pitch_do (o, input, out);
            pitchs.push_back(*(out->data));

//            aubio_mfcc_do(mfcc, fftgrain, mfcc_out);
            gammatone.startProcess(itePos, hop_s, gammaCoef);
            dctStep(counter, gammaCoef);
            ++counter;
            itePos += hop_s;

        }
        data.resize(NUMBER_OF_MFCC_FEATURES, counter);
        calcPitchData();

        del_aubio_pitch (o);
        del_fvec (out);
        del_fvec (input);
        fftw_free(fftwData);
        aubio_cleanup ();

        return 0;
    }


    void clear()
    {
        pitchs.clear();
        pitchData.clear();
        data.clear();
        resultData.clear();
        pitchNormalized.clear();
    }
};



#endif // DEFINITIONS_H
