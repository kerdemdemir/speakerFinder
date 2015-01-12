#ifndef PITCHGRAMS_H
#define PITCHGRAMS_H

#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>

class PitchGrams
{
public:

    PitchGrams(size_t gramCount, size_t stateCount)
    {
        m_gramCount = gramCount;
        m_stateCount = stateCount;
        totalCount = 0;
    }

    void Estimate(const std::vector<size_t>& input)
    {
        if (input.size() <= m_gramCount)
        {
            std::cout << "WARN <pitchGrams> Estimate: input size was less than input " << std::endl;
            return;
        }
        for (size_t i = 0; i < (input.size() - m_gramCount); i++)
        {
            size_t curHash = 0;
            for (size_t k = 0; k < m_gramCount; k++)
            {
                if (input[i] > m_stateCount || input[i + k] > m_stateCount )
                {
                    std::cout << "WARN <pitchGrams> Estimate: data was bigger state range " <<  input[i];
                    std::cout << " " << input[i + k] << std::endl;
                    continue;
                }
                curHash += (pow(100,k) * input[i + k]);
            }
            ngramCounts[curHash]++;
            totalCount++;
        }
    }

    void TrainingOver()
    {
        for (auto& elem : ngramCounts)
            ngramProbility[elem.first] = (double)elem.second / (double)totalCount;
    }

    double Probability(const std::vector<size_t>& input)
    {
        double totalProbility = 0.0;
        if (input.size() <= m_gramCount)
        {
            std::cout << "WARN <pitchGrams> Probibility: input size was less than input " << std::endl;
            return 0;
        }
        for (size_t i = 0; i < input.size() - m_gramCount; i++)
        {
            size_t curHash = 0;
            for (size_t k = 0; k < m_gramCount; k++)
            {
                curHash += (pow(100,k) * input[i + k]);
            }
            totalProbility += ngramProbility[curHash];
        }

        return totalProbility;
    }

    double Probability(const std::vector<size_t>::iterator inputIte,
                       const std::vector<size_t>::iterator endIte)
    {
        if (inputIte == (endIte - m_gramCount))
            return 0;
        size_t curHash = 0;
        for (size_t k = 0; k < m_gramCount; k++)
        {
            double curVal = *(inputIte + k);
            curHash += (pow(100,k) * curVal);
        }
        return ngramProbility[curHash];
    }


    void Save(int i)
    {
        std::string outputName = "PGram.txt";
        outputName += std::to_string(i);
        std::fstream outputFile(outputName.c_str(), std::ofstream::out);
        outputFile.precision(std::numeric_limits<double>::digits10);
        for (auto& elem : ngramProbility)
        {
            outputFile << elem.first << "," << elem.second << std::endl;
        }
        outputFile.close();
    }

    int Load(int i)
    {
        std::string inputName = "PGram.txt";
        inputName += std::to_string(i);
        std::ifstream input(inputName.c_str(), std::ifstream::binary);
        input.precision(std::numeric_limits<double>::digits10);
        int counter = 0;
        for (std::string line; std::getline(input, line); )
        {
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of(","));
            strs[1].pop_back();
            size_t key = std::stoi(strs[0]);
            double value = std::stod(strs[1]);
            ngramProbility[key] = value;
            counter++;
        }
        input.close();
        return counter;
    }

    size_t m_gramCount;
    size_t m_stateCount;
    std::unordered_map < size_t, size_t > ngramCounts;
    std::unordered_map < size_t, double > ngramProbility;
    size_t totalCount;


};

#endif // PITCHGRAMS_H
