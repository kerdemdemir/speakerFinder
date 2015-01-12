
#include <vector>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <definitions.h>
#include <trainer.h>


using namespace std;


Trainer trainer;
Gammatone gammatone;



int train( std::string& testFilePath)
{
    if (LOAD)
    {
        trainer.Load();
    }
    else
    {
        using namespace boost::filesystem;
        path p(testFilePath);
        directory_iterator iter(p), end;
        std::vector<std::pair< std::string, std::string> > fileNames;
        featureOutput localOutput;


        for(;iter != end; ++iter)
        {
            if (iter->path().extension() == ".wav")
            {
                std::string fullName = iter->path().string();
                std::string fileName = iter->path().filename().string();
                fileNames.push_back(std::make_pair(std::move(fullName), std::move(fileName)));
            }
        }

        for (auto& elem : fileNames)
        {
          if (localOutput.getFeatures(elem.first, gammatone) != -1)
            trainer.Estimate(elem.second, localOutput);
          localOutput.clear();
        }
    }
    std::cout << "Training is over " << std::endl;
}

void process( std::string& testFilePath, bool isTest)
{
    using namespace boost::filesystem;
    path p(testFilePath);
    directory_iterator iter(p), end;
    std::vector<std::pair< std::string, std::string> > fileNames;

    featureOutput localOutput;
    if (!isTest)
    {
        trainer.TrainingOver();
    }

    for(;iter != end; ++iter)
    {
        if (iter->path().extension() == ".wav")
        {
            std::string fullName = iter->path().string();
            std::string fileName = iter->path().filename().string();
            fileNames.push_back(std::make_pair(std::move(fullName), std::move(fileName)));
        }
    }

    for (auto& elem : fileNames)
    {
        if (localOutput.getFeatures(elem.first, gammatone) != -1)
            trainer.Probability(elem.second, localOutput, isTest);
        localOutput.clear();
    }
}



int main ()
{
    std::string trainPath("C:/Qt/Tools/QtCreator/bin/aubioSecond/train/cleaned");
    std::string validatePath("C:/Qt/Tools/QtCreator/bin/aubioSecond/test/validation/cleaned");
    std::string testPath("C:/Qt/Tools/QtCreator/bin/aubioSecond/test/test/cleaned");
    gammatone.init(64, 16000);

    train(trainPath);
    //Trainer.Save();
    process(validatePath, false);
    trainer.printValidationResults();
    process(testPath, true);
    trainer.printResults();

  /****************************** MACHINE LEARNINGGGGGGG *****************************///




  return 0;
}
