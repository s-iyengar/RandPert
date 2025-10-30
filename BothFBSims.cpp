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
	//Should (for set of perturbations) output n simulations
    //saving results {Exec time, sim time, steps, flux balance checks, cov balance checks, used parameters,avgs,variances+covariances,sensitivity_avg,sensitivityatavg}
    // Perturbation sims for the simplified or gate system
	string resultsfolder_og = "Data/Replicate_nofb2/" +return_current_time_and_date();
    fs::create_directories(resultsfolder_og);
    string delim = ",";
    std::vector<std::string> header_dat = {"Exec Time","Sim Time","Steps",
                                            "Flux Bal x","Flux Bal y",
                                            "Cov Bal xx","Cov Bal xy","Cov Bal yy",
                                            "lambda","beta_x","gamma","beta_y",
                                            "nx","Kx","ny","Ky",
                                            "<x>","<y>","Var(x)","Var(y)","Cov(xy)","<F_x>","<F_y>"};


    long int Nmax = 1e11;
	long int Nsteps = 5e6;

	state_t initialstate = {100,100};
	//{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset} however use positive hill coeffs: y is always neg fb and x is always pos fb

    std::vector<std::vector<double>> paramgen = {{1.},{5.},{3.},{1.},
                                                            {0.,0.,0.},{10.,20.,30.,40.,50.,60.},
                                                            {0.,0.,0.},{10.,20.,30.,40.,50.,60.},{0.1}};
    std::vector<std::vector<double>> baseparamlist = cart_product<double>(paramgen);
    

    std::vector<std::vector<std::vector<double>>> bigpertlist;
    std::vector<double> d_a_vals = {0.13590}; //log_space(-2,-0.3,10,5);
    std::vector<double> d_b_vals = {0.01920}; //log_space(-2,-0.3,10,5);
    
    std::vector<std::vector<double>> dnames(d_a_vals.size()*d_b_vals.size(),std::vector<double>(3));
    int itr1 = 0;
    //sqrt(N)=db/da
    for(double da:d_a_vals){
        for(double db:d_b_vals){
                dnames[itr1][0] = da;
                dnames[itr1][1] = db;
                dnames[itr1][2] = db*db/(da*da);
                itr1++;
                std::vector<std::vector<double>> pertlist = {{1.,1+da,1-da},{1.},{1.,1+db,1-db},{1.},
                                                            {1.},{1.},
                                                            {1.},{1.},
                                                            {1.}};
                std::vector<std::vector<double>> diff_perts = cart_product<double>(pertlist);

                std::vector<std::vector<double>> perts;
                for(std::vector<double> dp:diff_perts){
                    for(int i = 0;i<3;i++){
                        perts.push_back(dp);
                    }
                }
                bigpertlist.push_back(perts);
            }
        }
    #pragma omp parallel for num_threads(1) schedule(dynamic, 1)
    {
        for(int basenum = 0;basenum < baseparamlist.size();basenum++){
            stringstream sysfolder;
            sysfolder << "/system_" << basenum;
            string resultsfolder = resultsfolder_og + sysfolder.str();
            fs::create_directories(resultsfolder);

            for(int pertnum = 0;pertnum < bigpertlist.size();pertnum++){
                std::vector<std::vector<double>> perts = bigpertlist.at(pertnum);
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

                            //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky}
                            xy_posfx_or_negfy system{params.at(0),params.at(1),params.at(2),params.at(3),
                                                    params.at(4),params.at(5),params.at(6),params.at(7)};


                            std::random_device rd;
                            std::seed_seq sd{rd(), rd(), rd(), rd(),rd()};
                            rand_eng engine(sd);
                            
                            //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky}
                            covbal_fb_specifiiedposx_negfy alldata{params};
                            
                            auto begin = std::chrono::high_resolution_clock::now();
                            long int steps = gsim(system,alldata,initialstate,Nsteps,Nmax,engine);
                            auto end = std::chrono::high_resolution_clock::now();
                            auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
                            double exec_time = static_cast<double>(et.count());
                            double dsteps = static_cast<double>(steps);
                            covbaldata data = alldata.cbvals;
                            double w = data.wsum;
                            double zero = 0;

                            //saving results 
                            std::vector<double> streaming_data = {exec_time,w,dsteps,
                                                        rel_err(data.meanRpX,data.meanRmX),rel_err(data.meanRpY,data.meanRmY),
                                                        cov_balance_comp(data.CXRmX/w,data.CXRmX/w,data.meanRpX+data.meanRmX,data.CXRpX/w,data.CXRpX/w),
                                                        cov_balance_comp(data.CYRmX/w,data.CXRmY/w,0,data.CXRpY/w,data.CYRpX/w),
                                                        cov_balance_comp(data.CYRmY/w,data.CYRmY/w,data.meanRpY+data.meanRmY,data.CYRpY/w,data.CYRpY/w),
                                                        system.lambda,system.beta_x,system.gamma,system.beta_y,system.nx,system.Kx,
                                                        -1*system.ny,system.Ky,
                                                        data.meanx,data.meany,data.Vx/w,data.Vy/w,data.Cxy/w,
                                                        alldata.meanF_x,alldata.meanF_y};
                        allres.at(resrow) = (streaming_data);
                    }
                }
                ofstream results;
                    stringstream datapath;
                    datapath.precision(3);
                    datapath <<std::scientific<< resultsfolder << "/nums_"<< dnames[pertnum][2] << "n" << dnames[pertnum][0] << "n" << dnames[pertnum][1] << "n"<< basenum <<"_" << "data.csv";
                    results.open(datapath.str());
                    streamVector_endline<string>(results,header_dat,delim);
                    for(std::vector<double> streaming_data : allres){
                        streamVector_endline<double>(results,streaming_data,delim);
                    }
                results.close();
            }
        }
    }
	return 0;
}