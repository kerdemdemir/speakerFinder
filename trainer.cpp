#include "trainer.h"
#include <algorithm>

void Trainer::printResults()
{
    outputFile << "*************************" << std::endl;
    outputFile << "Final Resultsss  areeeee :" << std::endl;
    outputFile << "*************************" << std::endl;
    outputFile << "Correct Guess Gender" << correctGuessGender << " Total Guess " << totalGuess << std::endl;
    outputFile << "Percentage Gender" << (double)correctGuessGender / (double)totalGuess << std::endl;
    outputFile << "Correct Guess Speaker" << correctGuess << " Total Guess " << totalGuess << std::endl;
    outputFile << "Percentage Speaker" << (double)correctGuess / (double)totalGuess << std::endl;
    outputFile << "*************************" << std::endl;
    outputFile.close();
}

void Trainer::Probability(std::string& fileName, featureOutput& features, bool isTest)
{
    int state = fileName2State(fileName);
    std::fill(resultMFFC.begin(), resultMFFC.end(), 0);
    std::fill(resultPitch.begin(), resultPitch.end(), 0);
    std::fill(resultGrams.begin(), resultGrams.end(), 0);
    std::fill(resultGeneral.begin(), resultGeneral.end(), 0);


    for (size_t i = 0; i < features.data.n_cols; i++)
    {
        double propAllPeople = 0;
        double targetProbility  = 0;
        for (int k = 0; k < NUMBER_OF_PEOPLE; k++)
        {
            double tempProbility = gMFCCVec[k].Probability(features.data.col(i));
            propAllPeople += tempProbility;
            if (k == state)
                targetProbility = tempProbility;
            resultMFFC[k] += tempProbility;
        }
        features.pushResultData(targetProbility, propAllPeople, Method::MFCC);
    }

    for (size_t i = 0; i < features.pitchData.n_cols; i++)
    {
        double propAllPeople = 0;
        double targetProbility  = 0;
        for (int k = 0; k < NUMBER_OF_PEOPLE; k++)
        {
            double tempProbility = gPitchVec[k].Probability(features.pitchData.col(i));
            propAllPeople += tempProbility;
            if (k == state)
                targetProbility = tempProbility;
            resultPitch[k] += tempProbility;
        }
        features.pushResultData(targetProbility, propAllPeople, Method::PITCH);
    }

    for (size_t i = 0; i < features.pitchNormalized.size(); i++)
    {
        double propAllPeople = 0;
        double targetProbility  = 0;
        for (int k = 0; k < NUMBER_OF_PEOPLE; k++)
        {
            double tempProbility = gPitchGramsVec[k].Probability(features.pitchNormalized.begin() + i,
                                                                 features.pitchNormalized.end());
            propAllPeople += tempProbility;
            if (k == state)
                targetProbility = tempProbility;
            resultGrams[k] += tempProbility;
        }
        features.pushResultData(targetProbility, propAllPeople, Method::PGRAMS);
    }

    if (isTest)
    {
        for (size_t i = 0; i < features.resultData.n_cols; i++)
        {
            for (int k = 0; k < NUMBER_OF_PEOPLE; k++)
            {
                double tempProbility = gResultVec[k].Probability(features.resultData.col(i));
                resultGeneral[k] += tempProbility;
            }
        }
    }
    else
        gResultVec[state].Estimate(features.resultData);

    finalize(state, fileName, isTest);
}


void Trainer::finalize(int state, std::string& fileName,  bool isTest)
 {
     Method bestMethod = Method::PITCH;

     const char* genderStr = NULL ;
     double mfccSum =std::accumulate(resultMFFC.begin(),resultMFFC.end(),0.0);
     double pitchSum =std::accumulate(resultPitch.begin(),resultPitch.end(),0.0);
     double pGramsSum =std::accumulate(resultGrams.begin(),resultGrams.end(),0.0);
     double pGeneralSum =std::accumulate(resultGeneral.begin(),resultGeneral.end(),0.0);

     for (int k = 0; k < NUMBER_OF_PEOPLE; k++)
     {
         resultMFFC[k] = resultMFFC[k] / mfccSum;
         resultPitch[k] = resultPitch[k] / pitchSum;
         resultGrams[k] = resultGrams[k] / pGramsSum;
         resultGeneral[k] = resultGeneral[k] / pGeneralSum;
     }
     auto maxPitch = std::max_element(resultPitch.begin(), resultPitch.end());
     bool isFemale = !isMale(maxPitch);


     auto maxMFCC = isFemale ? std::max_element(resultMFFC.begin(), resultMFFC.end()) :
                        std::max_element(resultMFFC.begin()+3, resultMFFC.end());

     auto maxGrams = isFemale ? std::max_element(resultGrams.begin(), resultGrams.end()) :
                              std::max_element(resultGrams.begin()+3, resultGrams.end());

     auto maxGeneral = isFemale ? std::max_element(resultGeneral.begin(), resultGeneral.end()) :
                              std::max_element(resultGeneral.begin()+3, resultGeneral.end());

     if (isTest)
     {
         totalGuess++;
         auto pr = std::max_element(methodTruth[state].begin(), methodTruth[state].end(),
               [](const std::pair<Method, size_t>& p1, const std::pair<Method, size_t>& p2) {
                 return p1.second < p2.second; });
         bestMethod = pr->first;

         if (bestMethod == Method::MFCC)
         {
             genderStr = determineGender(resultMFFC.begin(), maxMFCC, state);
         }
         else if (bestMethod == Method::PITCH)
         {
             genderStr = determineGender(resultPitch.begin(), maxPitch, state);
         }
         else if (bestMethod == Method::PGRAMS)
         {
             genderStr = determineGender(resultGrams.begin(), maxGrams, state);
         }
         else if (bestMethod == Method::ALL_TRAINER)
         {
             genderStr = determineGender(resultGeneral.begin(), maxGrams, state);
         }
     }
     else
     {
         if (std::distance(resultMFFC.begin(), maxMFCC) == state)
         {
             methodTruth[state][Method::MFCC]++;
         }
         if (std::distance(resultPitch.begin(), maxPitch) == state)
         {
             methodTruth[state][Method::PITCH]++;
         }
         if (std::distance(resultGrams.begin(), maxGrams) == state)
         {
             methodTruth[state][Method::PGRAMS]++;
         }
         if (std::distance(resultGeneral.begin(), maxGeneral ) == state)
         {
             methodTruth[state][Method::ALL_TRAINER]++;
         }
     }



     outputFile << "*********************************************" << std::endl;
     outputFile <<  "The real excepted was state: " << state << "fileName " << fileName << std::endl;
     if (isTest)
     {
         outputFile <<  "Best method for this file is " << method2Str(bestMethod) << std::endl;
         if (genderStr != NULL)
             outputFile <<  "Gender is " << genderStr << std::endl;
     }

     outputFile <<  "MFCC Result was: " << std::distance(resultMFFC.begin(), maxMFCC) << std::endl;
     outputFile <<  "Pitch Result was: " << std::distance(resultPitch.begin(), maxPitch) << std::endl;
     outputFile <<  "PGRAM Result was: " << std::distance(resultGrams.begin(), maxGrams) << std::endl;
     outputFile <<  "Learner Result was: " << std::distance(resultGeneral.begin(), maxGeneral) << std::endl;

     outputFile << "*** All results ***" << std::endl;
     for (auto elem : resultMFFC)
         outputFile << elem << std::endl;

     outputFile << "*** Pitch ***" << std::endl;
     for (auto elem : resultPitch)
         outputFile << elem << std::endl;

     outputFile << "*** PGrams ***" << std::endl;
     for (auto elem : resultGrams)
         outputFile << elem << std::endl;

     outputFile << "*** Learner ***" << std::endl;
     for (auto elem : resultGeneral)
         outputFile << elem << std::endl;


     outputFile << "*********************************************" << std::endl;
     outputFile.flush();
 }


void Trainer::printValidationResults()
{
    std::cout << "Validation is over " << std::endl;
    outputFile << "Validation is over " << std::endl;
    for (size_t i = 0; i < methodTruth.size(); i++)
    {
        outputFile << "For person number " << (i+1) << std::endl;
        outputFile << "MFCC was correct for " <<methodTruth[i][Method::MFCC] << " times" << std::endl;
        outputFile << "PITCH was correct for " <<methodTruth[i][Method::PITCH] << " times" << std::endl;
        outputFile << "PGRAMS was correct for " <<methodTruth[i][Method::PGRAMS] << " times" << std::endl;
        outputFile << "Learner was correct for " <<methodTruth[i][Method::ALL_TRAINER] << " times" << std::endl;
    }
    std::cout << "**********************************" << std::endl;
}

const char*
Trainer::determineGender(std::vector <double>::iterator startPos, std::vector <double>::iterator maxPos, int state)
{
    size_t distance = std::distance(startPos, maxPos);
    if (distance == state)
     correctGuess++;

    if (state >= 3 &&  distance >= 3)
      correctGuessGender++;
    else if (state < 3 && distance < 3)
     correctGuessGender++;

    if (distance < 3)
     return "FEMALE";
    else
     return "MALE";
}


/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
void Trainer::Save()
{
    std::string mffcMeans = "mffcMean.txt";
    std::string pitchMeans = "pitchMean.txt";
    std::string mffcCovariance = "mffcCovariance.txt";
    std::string pitchCovariance = "pitchCovariance.txt";
    std::string mfccWeights = "mfccWeights.txt";
    std::string pitchWeights = "pitchWeights.txt";
    for (int k = 0; k < NUMBER_OF_PEOPLE; k++)
    {
        gPitchGramsVec[k].Save(k);
        std::string index = std::to_string(k);
        std::string mffcMeansName = mffcMeans + index;
        std::string pitchMeansName = pitchMeans + index;
        std::string mffcCovarianceName =  mffcCovariance + index;
        std::string pitchCovarianceName = pitchCovariance + index;
        std::string mfccWeightsName =  mfccWeights + index;
        std::string pitchWeightsName = pitchWeights + index;


        for (size_t i = 0; i < gMFCCVec[k].Means().size(); i++)
        {
            std::string indexInner = std::to_string(i);
            std::string mffcMeansInnerName = mffcMeansName + "_";
            mffcMeansInnerName += indexInner;
            arma::Col<double>& tempArma = gMFCCVec[k].Means()[i];
            tempArma.save(mffcMeansInnerName);
        }

        for (size_t i = 0; i < gPitchVec[k].Means().size(); i++)
        {
            std::string indexInner = std::to_string(i);
            std::string pitchMeansInnerName = pitchMeansName + "_";
            pitchMeansInnerName += indexInner;
            arma::Col<double>& tempArma = gPitchVec[k].Means()[i];
            tempArma.save(pitchMeansInnerName);
        }


        for (size_t i = 0; i < gMFCCVec[k].Covariances().size(); i++)
        {
            std::string indexInner = std::to_string(i);
            std::string mffcCovarianceInnerName = mffcCovarianceName + "_";
            mffcCovarianceInnerName += indexInner;
            arma::mat& tempArma = gMFCCVec[k].Covariances()[i];
            tempArma.save(mffcCovarianceInnerName);
        }

        for (size_t i = 0; i < gPitchVec[k].Covariances().size(); i++)
        {
            std::string indexInner = std::to_string(i);
            std::string pitchCovarianceInnerName = pitchCovarianceName + "_";
            pitchCovarianceInnerName += indexInner;
            arma::mat& tempArma = gPitchVec[k].Covariances()[i];
            tempArma.save(pitchCovarianceInnerName);
        }

        arma::Col<double>& tempArmaMFCCWeight = gMFCCVec[k].Weights();
        tempArmaMFCCWeight.save(mfccWeightsName);


        arma::Col<double>& tempArmaPitchWeight = gMFCCVec[k].Weights();
        tempArmaPitchWeight.save(pitchWeightsName);
    }
}

void Trainer::Load()
{
    std::string mffcMeans = "mffcMean.txt";
    std::string pitchMeans = "pitchMean.txt";
    std::string mffcCovariance = "mffcCovariance.txt";
    std::string pitchCovariance = "pitchCovariance.txt";
    std::string mfccWeights = "mfccWeights.txt";
    std::string pitchWeights = "pitchWeights.txt";
    for (int k = 0; k < NUMBER_OF_PEOPLE; k++)
    {
        gPitchGramsVec[k].Load(k);

        std::string index = std::to_string(k);
        std::string mffcMeansName = mffcMeans + index;
        std::string pitchMeansName = pitchMeans + index;
        std::string mffcCovarianceName =  mffcCovariance + index;
        std::string pitchCovarianceName = pitchCovariance + index;
        std::string mfccWeightsName =  mfccWeights + index;
        std::string pitchWeightsName = pitchWeights + index;

        size_t i = 0;
        while(true)
        {
            std::string indexInner = std::to_string(i);
            std::string mffcMeansInnerName = mffcMeansName + "_";
            mffcMeansInnerName += indexInner;
            arma::Col<double> tempArma;
            if (!tempArma.load(mffcMeansInnerName))
                break;
            gMFCCVec[k].Means()[i] = (std::move(tempArma));
            i++;
        }

        i = 0;
        while(true)
        {
            std::string indexInner = std::to_string(i);
            std::string pitchMeansInnerName = pitchMeansName + "_";
            pitchMeansInnerName += indexInner;
            arma::Col<double> tempArma;
            if (!tempArma.load(pitchMeansInnerName))
                break;
            gPitchVec[k].Means()[i] = std::move(tempArma);
            i++;
        }

        i = 0;
        while(true)
        {
            std::string indexInner = std::to_string(i);
            std::string mffcCovarianceInnerName = mffcCovarianceName + "_";
            mffcCovarianceInnerName += indexInner;
            arma::mat tempArma;
            if (!tempArma.load(mffcCovarianceInnerName))
                break;
            gMFCCVec[k].Covariances()[i] = std::move(tempArma);
            i++;
        }

        i = 0;
        while(true)
        {
            std::string indexInner = std::to_string(i);
            std::string pitchCovarianceInnerName = pitchCovarianceName + "_";
            pitchCovarianceInnerName += indexInner;
            arma::mat tempArma;
            if(!tempArma.load(pitchCovarianceInnerName))
                break;
            gPitchVec[k].Covariances()[i] = std::move(tempArma);
            i++;
        }

        arma::Col<double> tempArmaMFCCWeight;
        tempArmaMFCCWeight.load(mfccWeightsName);
        gMFCCVec[k].Weights() = std::move(tempArmaMFCCWeight);

        arma::Col<double> tempArmaPitchWeight;
        tempArmaMFCCWeight.load(pitchWeightsName);
        gPitchVec[k].Weights() = std::move(tempArmaPitchWeight);
    }
}
