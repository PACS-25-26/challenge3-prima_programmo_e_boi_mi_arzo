#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <json.hpp>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

// --- Helper function to load the json file ---
static json load(const std::string& fileName){

    // Check if file is present and correctly open
    std::ifstream input_file(fileName);
    if (!input_file)
        std::cerr << "Cannot open "<<fileName;

    // Write on the json
    const json inputJson = json::parse(input_file);

    return inputJson;
}


// --- STRUCT PARAMETERS ---
struct Parameters{
    const double bcTop;
    const double bcBottom;
    const double bcLeft;
    const double bcRight;
    const unsigned N;
    const double tol;
    const unsigned maxIt;

/* 
   In the following we use a public constructor that makes a call to the
   one to be able to initialize the const parameters reading them from the json file.
*/

    // Public constructor with file name that loads the json and calls the other constructor
    Parameters(const std::string& fileName) : 
        Parameters(load(fileName)){};

    // Real constructor
    Parameters(const json& inputJson) :
        bcTop(inputJson["bcTop"]),
        bcBottom(inputJson["bcBottom"]),
        bcLeft(inputJson["bcLeft"]),
        bcRight(inputJson["bcRight"]),
        N(inputJson["N"]),
        tol(inputJson["tol"]),
        maxIt(inputJson["maxIt"])
    {
        // Correctly read the parameters
        std::cout<<"Correctly read the json file and set all the parameters"<<std::endl;
    };
};



#endif