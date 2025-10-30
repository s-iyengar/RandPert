#ifndef PERTUTILS
#define PERTUTILS

#include "System.hpp"
#include "DataStorage.hpp"
#include "Gillespie.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <regex>
#include <algorithm>


//Based on https://stackoverflow.com/a/17050528
template<typename T>
std::vector<std::vector<T>> cart_product (const std::vector<std::vector<T>>& factors) {
    std::vector<std::vector<T>> s = {{}};
    for (const auto& factor : factors) {
        std::vector<std::vector<T>> r;
        for (const auto& x : s) {
            for (const auto y : factor) {
                r.push_back(x);
                r.back().push_back(y);
            }
        }
        s = move(r);
    }
    return s;
}

std::vector<double> linspace(double start, double stop, int N){
    double a = (stop-start)/N;
    std::vector<double> v(N);
    std::generate(v.begin(), v.end(), [n = 0, &a,&start]() mutable { return start + (n++ * a); });
    return v;
}

std::vector<double> pertexperiment_orgate(std::vector<double> const & pert, std::vector<double> baseparams,state_t initialstate,long int Nsteps,long int Nmax){
            std::vector<double> params = baseparams;
            for(int i=0; i < pert.size(); i++){
                params.at(i) *= pert.at(i);
            }
            //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset}
            xy_xORyFB_shared system{params.at(0),params.at(1),params.at(2),params.at(3),
                                    params.at(4),params.at(5),params.at(6),params.at(7),
                                    params.at(8)};


            std::random_device rd;
            std::seed_seq sd{rd(), rd(), rd(), rd(),rd()};
            rand_eng engine(sd);
            
            //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset}
            covbaldata data;
            
            auto begin = std::chrono::high_resolution_clock::now();
            long int steps = gsim(system,data,initialstate,Nsteps,Nmax,engine);
            auto end = std::chrono::high_resolution_clock::now();
            auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
            double exec_time = static_cast<double>(et.count());
            double dsteps = static_cast<double>(steps);
            double w = data.wsum;

            //saving results {Exec time, sim time, steps, flux balance checks, cov balance checks, used parameters,avgs,variances+covariances,sensitivity_avg,sensitivityatavg}
            std::vector<double> streaming_data = {exec_time,w,dsteps,
                                        rel_err(data.meanRpX,data.meanRmX),rel_err(data.meanRpY,data.meanRmY),
                                        cov_balance_comp(data.CXRmX/w,data.CXRmX/w,data.meanRpX+data.meanRmX,data.CXRpX/w,data.CXRpX/w),
                                        cov_balance_comp(data.CYRmX/w,data.CXRmY/w,0,data.CXRpY/w,data.CYRpX/w),
                                        cov_balance_comp(data.CYRmY/w,data.CYRmY/w,data.meanRpY+data.meanRmY,data.CYRpY/w,data.CYRpY/w),
                                        params.at(0),params.at(1),params.at(2),params.at(3),
                                        params.at(4),params.at(5),params.at(6),
                                        params.at(7),params.at(8),
                                        data.meanx,data.meany,data.Vx/w,data.Vy/w,data.Cxy/w,
                                        ORgate_F_x<double>(data.meanx,data.meany,params.at(0),params.at(4),params.at(6),params.at(5),params.at(7),params.at(8)),
                                        ORgate_F_y<double>(data.meanx,data.meany,params.at(0),params.at(4),params.at(6),params.at(5),params.at(7),params.at(8))};
    return streaming_data;
}

std::vector<double> log_space(double firstexp, double lastexp, double base, int k) {
    //generate list {base^firstexp ... base^lastexp} with linearly spaced exps k
  std::vector<double> logspace = linspace(firstexp,lastexp,k);
  std::for_each(logspace.begin(), logspace.end(),
                [base](double &x) { x = std::pow(base, x); });
  return logspace;
}


const std::regex delimiter{ "," };

std::vector<std::vector<double>> system_csv_read(std::string csvFileName) {
    //Based on https://stackoverflow.com/a/60326675
    // Open the file and check, if it could be opened
    std::vector<std::vector<double>> numericData;
    if (std::ifstream csvFile(csvFileName); csvFile) {
        // This is our "2D array string vector" as described in your post
        std::vector<std::vector<std::string>> csvData{};
        // Read the complete CSV FIle into a 2D vector ----------------------------------------------------
        // We will read all lines of the source file with a simple for loop and std::getline
        for (std::string line{}; std::getline(csvFile, line); ) {
            // We will split the one big string into tokens (sub-strings) and add it to our 2D array
            csvData.emplace_back(std::vector<std::string>(std::sregex_token_iterator(line.begin(), line.end(), delimiter, -1), {}));
        }
        // -------------------------------------------------------------------------------------------------
        //String vector to double https://stackoverflow.com/a/70458777

        std::transform(csvData.begin(), csvData.end(), std::back_inserter(numericData), [](const auto& strs) {
        std::vector<double> result;
        std::transform(strs.begin(), strs.end(), std::back_inserter(result), [](const auto& str) { return std::stod(str); });
        return result;
                        });
    }
    return numericData;
}

template<typename dist>
std::vector<double> random_vector(int size,dist distribution){
    
    std::vector<double> randvec(size);

    std::random_device rd;
    std::seed_seq sd{rd(), rd(), rd(), rd(),rd()};
    rand_eng engine(sd);

    std::generate(randvec.begin(), randvec.end(),
                 [&](){ return distribution(engine); });

    return randvec;
}

#endif