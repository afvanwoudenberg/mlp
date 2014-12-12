// xor.cpp
// Aswin van Woudenberg
// Compile with: g++ -std=c++11 -o xor xor.cpp

#include "mlp.h"
#include <tuple>
#include <vector>
#include <ctime>
#include <iostream>

int main()
{
    std::srand(std::time(0));
    
    Bpn<2,2,1> bpn;
  
    std::vector<std::tuple<double,double>> input = { 
        std::make_tuple(0.0,0.0), 
        std::make_tuple(1.0,0.0), 
        std::make_tuple(0.0,1.0), 
        std::make_tuple(1.0,1.0) 
    };
    std::vector<double> output = {0, 1, 1, 0};
 
    // bpn.load("xor.nn");
    bpn.train(input, output, 10000, 0.0);
    for (int i = 0; i < input.size(); ++i)
    {
        std::tuple<double,double> &tuple = input[i];
        bpn.ff(tuple);
        std::cout << std::get<0>(tuple) << " " << std::get<1>(tuple) << " " 
                  << output[i] << " " << bpn.mlp.nodes[0].output << std::endl;
    }
    std::cout << "Overall error: " << bpn.overallError << std::endl;
    bpn.save("xor.nn");

    return 0;
}

