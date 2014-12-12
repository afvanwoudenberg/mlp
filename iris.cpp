// iris.cpp
// Aswin van Woudenberg
// Compile with: g++ -std=c++11 -o iris iris.cpp

#include "mlp.h"
#include <tuple>
#include <vector>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

void loadIris(const std::string &filename, 
    std::vector<std::tuple<double,double,double,double>> &input, 
    std::vector<std::tuple<double,double,double>> &output)
{
    double sepalLength, sepalWidth, petalLength, petalWidth, // input
        irisSetosa, irisVersicolour, irisVirginica; // output

    std::ifstream ifs(filename);
    while (ifs.good()) {
        ifs >> sepalLength >> sepalWidth >> petalLength >> petalWidth >>
            irisSetosa >> irisVersicolour >> irisVirginica;
        input.push_back(std::make_tuple(sepalLength, sepalWidth, petalLength, petalWidth));
        output.push_back(std::make_tuple(irisSetosa, irisVersicolour, irisVirginica));
    }
    ifs.close();
}

int main()
{
    std::srand(std::time(0));

    Bpn<4,7,3> bpn;

    std::vector<std::tuple<double,double,double,double>> inputTrain;
    std::vector<std::tuple<double,double,double>> outputTrain;

    std::vector<std::tuple<double,double,double,double>> inputTest;
    std::vector<std::tuple<double,double,double>> outputTest;

    std::cout << "Reading training and test set..." << std::endl;
    loadIris("iris_train.dat", inputTrain, outputTrain);
    loadIris("iris_test.dat", inputTest, outputTest);
    
    // bpn.load("iris.nn");
    std::cout << "Training the neural network..." << std::endl;
    bpn.train(inputTrain, outputTrain, 2000, 0.0, 0.05, 0.01);

    std::cout << "Calculating correctly guessed Iris plants..." << std::endl;
    int guessedCorrectly = 0;
    for (int i = 0; i < inputTest.size(); ++i)
    {
        std::tuple<double,double,double,double> &tuple = inputTest[i];
        bpn.ff(tuple);
        if (fabs(std::get<0>(outputTest[i]) - bpn.mlp.nodes[0].output) < 0.5 &&
            fabs(std::get<1>(outputTest[i]) - bpn.mlp.nodes[1].output) < 0.5 &&
            fabs(std::get<2>(outputTest[i]) - bpn.mlp.nodes[2].output) < 0.5)
            guessedCorrectly++;
    }
    std::cout << "Guessed correctly: " << static_cast<double>(guessedCorrectly * 100)/inputTest.size() << "%" << std::endl;
    std::cout << "Saving neural net weights..." << std::endl;
    bpn.save("iris.nn");
    std::cout << "Done." << std::endl;

    return 0;
}

