#include "Gillespie.hpp"
#include "DataStorage.hpp"
#include "System.hpp"
#include "OutputUtils.hpp"
#include "PertUtils.hpp"


#include <random>
#include <algorithm>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <ctime>
#include <ios>

struct hybrid{
    vector_map_unordered pdist;
    covbal_fb_sharedor cbvals;
    hybrid(std::vector<double> pars){
        cbvals = covbal_fb_sharedor(pars);
    }   
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        pdist.recordData(currentstate,rates,w);
    }
};

int main(){
    std::vector<double> pars = {5.0,0.1,0.6,1.0,5.5,40.0,-4.0,20.0};
    state_t initialstate = {100,100};
    long int Nmax = 1e11;
    long int Nsteps = 5e6;
    xy_xORyFB_shared system{pars[0],pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],pars[7],pars[8]};


    std::random_device rd;
    std::seed_seq sd{rd(), rd(), rd(), rd(),rd()};
    rand_eng engine(sd);
    
    
    hybrid bothdata(pars);
    auto begin = std::chrono::high_resolution_clock::now();
    long int steps = gsim(system,bothdata,initialstate,Nsteps,Nmax,engine);
    auto end = std::chrono::high_resolution_clock::now();
    auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);

    std::unordered_map<state_t,precision> normalised = bothdata.pdist.normaliseHist();

    std::cout << et.count() << std::endl;
    std::vector<state_t> states;
    states.reserve(normalised.size());
    std::vector<precision> vals;
    vals.reserve(normalised.size());

    for(auto kv : normalised) {
        states.push_back(kv.first);
        vals.push_back(kv.second);  
        }

    std::ofstream outfile;
    outfile.open("test.csv");
    std::vector<std::string> header = {"x","y","p"};
    streamVector_endline(outfile,header,",");
    for(int i = 0;i<normalised.size();i++){
        streamVector_contline(outfile,states[i],",");
        outfile << vals[i] << std::endl;
    }
    outfile.close();
    covbal_fb_sharedor dat = bothdata.cbvals;
    covbaldata data = dat.cbvals;
    double w = data.wsum;
    double zero = 0;

    //saving results {Exec time, sim time, steps, flux balance checks, cov balance checks, used parameters,avgs,variances+covariances,sensitivity_avg,sensitivityatavg}
    std::vector<std::string> titles = {"Exec time","sim time","steps","flux bal x","flux bal y","cov bal xx","cov bal xy","cov bal yy","lambda","beta_x","gamma","beta_y","nx","Kx","ny","Ky","offset","meanx","meany","Vx","Vy","Cxy","meanF_x","meanF_y"};
    std::vector<double> streaming_data = {(double)et.count(),w,(double)steps,
                                                        rel_err(data.meanRpX,data.meanRmX),rel_err(data.meanRpY,data.meanRmY),
                                                        cov_balance_comp(data.CXRmX/w,data.CXRmX/w,data.meanRpX+data.meanRmX,data.CXRpX/w,data.CXRpX/w),
                                                        cov_balance_comp(data.CYRmX/w,data.CXRmY/w,0,data.CXRpY/w,data.CYRpX/w),
                                                        cov_balance_comp(data.CYRmY/w,data.CYRmY/w,data.meanRpY+data.meanRmY,data.CYRpY/w,data.CYRpY/w),
                                                        system.lambda,system.beta_x,system.gamma,system.beta_y,system.nx,system.Kx,
                                                        system.ny,system.Ky,system.offset,
                                                        data.meanx,data.meany,data.Vx/w,data.Vy/w,data.Cxy/w,
                                                        dat.meanF_x,dat.meanF_y};
    std::ofstream twofile;
    std::vector<std::string> titles2 = {"MeanRpX","MeanRmX","MeanRpY","MeanRmY",
                                        "CXRmX","CXRmY","CXRpX","CXRpY",
                                        "CYRmX","CYRmY","CYRpX","CYRpY"};    
    std::vector<double> streaming_data_2 = {data.meanRpX,data.meanRmX,data.meanRpY,data.meanRmY,
                                            data.CXRmX/w,data.CXRmY/w,data.CXRpX/w,data.CXRpY/w,
                                            data.CYRmX/w,data.CYRmY/w,data.CYRpX/w,data.CYRpY/w};
    twofile.open("test2.csv");
    streamVector_endline(twofile,titles,",");
    streamVector_endline(twofile,streaming_data,",");
    streamVector_endline(twofile,titles2,",");
    streamVector_endline(twofile,streaming_data_2,",");
    twofile.close(); 
    return 0;
}