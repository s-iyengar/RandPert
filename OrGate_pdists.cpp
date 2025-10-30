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

struct covbal_withpdist{
    covbal_fb_sharedor cbvals;
    std::vector<double> params; 
    vector_map_unordered pdist;
    covbal_withpdist(std::vector<double> pars){
        params = pars;
        cbvals = covbal_fb_sharedor{params};
        pdist = vector_map_unordered();
    }
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        pdist.recordData(currentstate,rates,w);
    }
};

int main(){
	//Should (for set of perturbations) output n simulations
    //saving results {Exec time, sim time, steps, flux balance checks, cov balance checks, used parameters,avgs,variances+covariances,sensitivity_avg,sensitivityatavg}
    // Perturbation sims for the simplified or gate system 
	string resultsfolder_og = "Data/fx_violations_withpdist/" +return_current_time_and_date();
    fs::create_directories(resultsfolder_og);
    string delim = ",";
    std::vector<std::string> header_dat = {"Exec Time","Sim Time","Steps",
                                            "Flux Bal x","Flux Bal y",
                                            "Cov Bal xx","Cov Bal xy","Cov Bal yy",
                                            "lambda","beta_x","gamma","beta_y",
                                            "nx","Kx","ny","Ky","offset",
                                            "<x>","<y>","Var(x)","Var(y)","Cov(xy)","<F_x>","<F_y>"};


    long int Nmax = 1e11;
	long int Nsteps = 1e6;
    std::vector<std::string> header_pdist = {"x","y","p"};
	state_t initialstate = {100,100};
	//{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset} 5,0.1,0.6,1,5.5,40,-3,20,0

    std::vector<std::vector<double>> baseparamlist = {{50.0,1.0,6.0,10.0,3.5,40,-4,20,0.0},
                                                {5.0,0.1,0.6,1.0,3.5,40,-5,20,0.0},
                                                {5.0,0.1,0.6,1.0,4.0,40,-5,20,0.0},
                                                {5.0,0.1,0.6,1.0,4.0,40,-4,20,0.0},
                                                {5.0,0.1,0.6,1.0,3.5,40,-4,20,0.0},
                                                {3.75,0.3,2.3,1.0,4.0,10,-1,10,0.},
                                                {3.75,0.3,2.3,1.0,4.2,10,-1,10,0.},
                                                {3.75,0.3,2.3,1.0,3.8,10,-1,10,0.},
                                                {3.75,0.3,2.3,1.0,3.6,10,-1,10,0.},};
    

    std::vector<std::vector<std::vector<double>>> bigpertlist;
    std::vector<double> d_a_vals = log_space(-2,-0.05,10,3);
    std::vector<double> d_b_vals = log_space(-1,-0.01,10,3);
    
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

                    #pragma omp parallel for shared(allres) num_threads(1) schedule(dynamic,1)
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
                                    covbal_withpdist alldata{params};
                                    
                                    auto begin = std::chrono::high_resolution_clock::now();
                                    long int steps = gsim(system,alldata,initialstate,Nsteps,Nmax,engine);
                                    auto end = std::chrono::high_resolution_clock::now();
                                    auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
                                    double exec_time = static_cast<double>(et.count());
                                    double dsteps = static_cast<double>(steps);
                                    covbaldata data = alldata.cbvals.cbvals;
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
                                                                alldata.cbvals.meanF_x,alldata.cbvals.meanF_y};
                                allres.at(resrow) = (streaming_data);
                                std::unordered_map<state_t,precision> normalised = alldata.pdist.normaliseHist();
                                ofstream pdist_file;
                                    stringstream pdistpath;
                                    pdistpath.precision(3);
                                    pdistpath <<std::scientific<< resultsfolder << "/da_" << dnames[pertnum][0] << "_db_" << dnames[pertnum][1] << "_sys_"<< basenum <<"_pert_"<<resrow << "_pdist.csv";
                                    pdist_file.open(pdistpath.str());
                                    streamVector_endline<string>(pdist_file,header_pdist,delim);
                                    for(auto const& [key, val] : normalised){
                                        pdist_file << key[0] << "," << key[1] << "," << val << std::endl;
                                    }
                                pdist_file.close();
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