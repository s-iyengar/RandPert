#ifndef OUTPUTUTILS
#define OUTPUTUTILS

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

template<typename T>
void streamVector_contline(std::ofstream& file,std::vector<T> datavec,std::string delim){
    for(T i:datavec){
        file << i << delim;
    }
}

template<typename T>
void streamVector_endline(std::ofstream& file,std::vector<T> datavec,std::string delim){
    for(int i = 0;i<datavec.size()-1;i++){
        file << datavec.at(i) << delim;
    }
    file << datavec.at(datavec.size()-1)<< std::endl;
}

template<typename T>
void writeVetorofVectors(std::ofstream& file,std::vector<std::vector<T>> datavectors,std::string delim){
    for(std::vector<T> datavec:datavectors){
        writeVectorLine(file,datavec,delim);
        file << std::endl;
    }
}

template<typename T>
void stream_element(std::ofstream& file,T element,std::string delim){
    file << element << delim;
}


#endif