#ifndef TRAINER_H
#define TRAINER_H

#include <map>
#include <pitchgrams.h>
#include <definitions.h>

class Trainer
{
public:

    Trainer()  : gMFCCVec(NUMBER_OF_PEOPLE, GMM<>(5 ,NUMBER_OF_MFCC_FEATURES)),
                 gPitchVec(NUMBER_OF_PEOPLE, GMM<>(5,1)),
                 gResultVec(NUMBER_OF_PEOPLE, GMM<>(3,3)),
                 gPitchGramsVec(NUMBER_OF_PEOPLE, PitchGrams(4, STATE_COUNT))
    {
        methodTruth.resize(NUMBER_OF_PEOPLE);
        resultMFFC.resize(NUMBER_OF_PEOPLE);
        resultPitch.resize(NUMBER_OF_PEOPLE);
        resultGrams.resize(NUMBER_OF_PEOPLE);
        resultGeneral.resize(NUMBER_OF_PEOPLE);
        outputFile.open("machineLogging.txt", std::ofstream::out);
    }

    void Estimate(std::string& fileName, featureOutput& features)
    {
        int state = fileName2State(fileName);

        gMFCCVec[state].Estimate(features.data);
        gPitchVec[state].Estimate(features.pitchData);
        gPitchGramsVec[state].Estimate(features.pitchNormalized);
    }

    void TrainingOver()
    {
        for (size_t i = 0; i < gPitchGramsVec.size(); i++)
        {
            gPitchGramsVec[i].TrainingOver();
        }
        if (!LOAD)
            Save();
    }

    void Probability(std::string& fileName, featureOutput& features, bool isTest);
    void printValidationResults();
    void finalize(int state, std::string& fileName,  bool isTest);
    void printResults();
    void Save();
    void Load();


    private:

    size_t fileName2State(std::string& fileName)
    {
        size_t state = 0;
        if (fileName[0] == 'm')
            state = 3;
        state += (fileName[1] - 49);
        return state;
    }

    bool
    isMale(std::vector <double>::iterator maxPos)
    {
        size_t distance = std::distance(resultPitch.begin(), maxPos);
        if (distance < 3)
            return false;
        else
            return true;
    }

    const char*
    determineGender(std::vector <double>::iterator startPos, std::vector <double>::iterator maxPos, int state);

    std::vector< GMM<> > gMFCCVec;
    std::vector< GMM<> > gPitchVec;
    std::vector< GMM<> > gResultVec;
    std::vector< PitchGrams > gPitchGramsVec;
    std::vector <double> resultMFFC;
    std::vector <double> resultPitch;
    std::vector <double> resultGrams;
    std::vector <double> resultGeneral;
    std::vector < std::map<Method, size_t> >    methodTruth;
    size_t correctGuess;
    size_t correctGuessGender;
    size_t totalGuess;
    std::fstream outputFile;
};

#endif // TRAINER_H
