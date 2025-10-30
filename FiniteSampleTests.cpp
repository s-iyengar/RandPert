#include "Gillespie.hpp"
#include "DataStorage.hpp"
#include "System.hpp"
#include "OutputUtils.hpp"
#include "PertUtils.hpp"

#include <omp.h>

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

using namespace std;
namespace fs = std::filesystem;



int main(){
	//Should (for set of perturbations) output n simulation
    //saving results {Exec time, sim time, steps, flux balance checks, cov balance checks, used parameters,avgs,variances+covariances,sensitivity_avg,sensitivityatavg}
    // Perturbation sims for the simplified or gate system
	string resultsfolder_og = "FiniteSampleTest/" +return_current_time_and_date();
    fs::create_directories(resultsfolder_og);
    omp_set_max_active_levels(3);
    string delim = ",";
    std::vector<std::string> header_dat = {"Exec Time","Sim Time","Steps",
                                            "Flux Bal x","Flux Bal y",
                                            "Cov Bal xx","Cov Bal xy","Cov Bal yy",
                                            "lambda","beta_x","gamma","beta_y",
                                            "nx","Kx","ny","Ky","offset",
                                            "<x>","<y>","Var(x)","Var(y)","Cov(xy)","<F_x>","<F_y>"};


    long int Nmax = 1e11;
	long int Nsteps = 1e6;

	state_t initialstate = {100,100};
	//{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset}

    std::vector<int> num_samples = {5,10,25,50,100,150,200,300,400};

    

    std::vector<std::vector<double>> paramgen = {{5.},{5.,1.},{10.},{10.,1.},
                                                            {0.},{1.},
                                                            {-1.},{1.,100.},
                                                            {0.}};
    std::vector<std::vector<double>> baseparamlist = cart_product<double>(paramgen);
    
    std::uniform_real_distribution<double> lambda_dist(-0.5,0.5);
    std::uniform_real_distribution<double> gamma_dist(-0.5,0.5);
    std::random_device pd;
    std::seed_seq psd{pd(), pd(), pd(), pd(),pd()};
    rand_eng pengine(psd);


        for(int basenum = 0;basenum < baseparamlist.size();basenum++){
            string resultsfolder = resultsfolder_og + "/system_" + to_string(basenum);
            fs::create_directories(resultsfolder);
            #pragma omp parallel for num_threads(2) schedule(dynamic, 1)
            {
            for(int n_ind = 0;n_ind < num_samples.size();n_ind++){
                int samples = num_samples.at(n_ind);
                std::vector<std::vector<double>> perts(samples,std::vector<double>(9));
                for (int i = 0; i < samples; i++){
                    perts.at(i) = {1+lambda_dist(pengine),1,1+gamma_dist(pengine),1,1,1,1,1,1};
                }
                std::vector<double> baseparams = baseparamlist.at(basenum);
                //Vectors to store results
                std::vector<std::vector<double>> allres(perts.size(),std::vector<double>(header_dat.size()));

                #pragma omp parallel for shared(allres) num_threads(3) schedule(dynamic,1)
                {
                    for(int resrow = 0;resrow < perts.size();resrow++){
                        std::vector<double> pert = perts.at(resrow);
                        std::vector<double> params = baseparams;
                        for(int i=0; i < pert.size(); i++){
                                if(pert.at(i) != 1){
                                    params.at(i) *= pert.at(i);
                                }
                            }

                            //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset}
                            xy_xORyFB_shared system{params.at(0),params.at(1),params.at(2),params.at(3),
                                                    params.at(4),params.at(5),params.at(6),params.at(7),
                                                    params.at(8)};


                            std::random_device rd;
                            std::seed_seq sd{rd(), rd(), rd(), rd(),rd()};
                            rand_eng engine(sd);
                            
                            //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset}
                            covbal_fb_sharedor alldata{params};
                            
                            auto begin = std::chrono::high_resolution_clock::now();
                            long int steps = gsim(system,alldata,initialstate,Nsteps,Nmax,engine);
                            auto end = std::chrono::high_resolution_clock::now();
                            auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
                            double exec_time = static_cast<double>(et.count());
                            double dsteps = static_cast<double>(steps);
                            covbaldata data = alldata.cbvals;
                            double w = data.wsum;
                            double zero = 0;

                            //saving results {Exec time, sim time, steps, flux balance checks, cov balance checks, used parameters,avgs,variances+covariances,sensitivity_avg,sensitivityatavg}
                            std::vector<double> streaming_data = {exec_time,w,dsteps,
                                                        rel_err(data.meanRpX,data.meanRmX),rel_err(data.meanRpY,data.meanRmY),
                                                        cov_balance_comp(data.CXRmX/w,data.CXRmX/w,data.meanRpX+data.meanRmX,data.CXRpX/w,data.CXRpX/w),
                                                        cov_balance_comp(data.CYRmX/w,data.CXRmY/w,0,data.CXRpY/w,data.CYRpX/w),
                                                        cov_balance_comp(data.CYRmY/w,data.CYRmY/w,data.meanRpY+data.meanRmY,data.CYRpY/w,data.CYRpY/w),
                                                        system.lambda,system.beta_x,system.gamma,system.beta_y,system.nx,system.Kx,
                                                        system.ny,system.Ky,system.offset,
                                                        data.meanx,data.meany,data.Vx/w,data.Vy/w,data.Cxy/w,
                                                        alldata.meanF_x,alldata.meanF_y};
                        allres.at(resrow) = (streaming_data);
                    }
                }
                ofstream results;
                    stringstream datapath;
                    datapath.precision(3);
                    datapath << resultsfolder <<"/system_" << basenum<<"-"<<"N"<< samples << "-"<< "data.csv";
                    results.open(datapath.str());
                    streamVector_endline<string>(results,header_dat,delim);
                    for(std::vector<double> streaming_data : allres){
                        streamVector_endline<double>(results,streaming_data,delim);
                    }
                results.close();
                ofstream metadata;
                    stringstream metapath;
                    metapath.precision(3);
                    metapath << resultsfolder <<"/system_" << basenum<<"-"<<"N"<< samples << "-"<< "metadata.csv";
                    metadata.open(metapath.str());
                    metadata << "lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset" << endl;
                    streamVector_endline<double>(metadata,baseparams,delim);
                    for(std::vector<double> pert : perts){
                        streamVector_endline<double>(metadata,pert,delim);
                    }
                    metadata.close();
            }
        }
    }
	return 0;
}