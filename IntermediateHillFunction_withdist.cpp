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
    covbal_inthill cbvals;
    std::vector<double> params; 
    vector_map_unordered pdist;
    covbal_withpdist(std::vector<double> pars){
        params = pars;
        cbvals = covbal_inthill{params};
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
	string resultsfolder_og = "Data/IntHill_indirect_comp/" +return_current_time_and_date();
    fs::create_directories(resultsfolder_og);
    omp_set_max_active_levels(2);
    string delim = ",";
    std::vector<std::string> header_dat = {"Exec Time","Sim Time","Steps",
                                            "Flux Bal x","Flux Bal y","Flux Bal z",
                                            "Cov Bal xx","Cov Bal xy","Cov Bal yy",
                                            "Cov Bal xz","Cov Bal yz","Cov Bal zz",
                                            "lambda","beta_x","gamma","beta_y","alpha","beta_z",
                                            "nx","Kx","ny","Ky","offset",
                                            "<x>","<y>","<z>",
                                            "Var(x)","Var(y)","Var(z)","Cov(xy)","Cov(xz)","Cov(yz)",
                                            "<F_x>","<F_y>"};

    std::vector<std::string> header_pdist = {"x","y","z","p"};
    long int Nmax = 1e10;
	long int Nsteps = 1e7;
    std::uniform_int_distribution<int> initial_state_dist(1,200); // To start each simulation independently
	
	//{lambda,beta_x,gamma,beta_y,alpha,beta_znx,Kx,ny,Ky,offset} however use positive hill coeffs: y is always neg fb and x is always pos fb
    std::vector<std::vector<double>> paramgen = {{4.5},{0.1}, //lambda beta_x
                                                {0.7},{1.}, //gamma beta_y
                                                {1.},{1.}, //alpha beta_z
                                                {6},{40.}, //nx Kx
                                                {-5.},{20.}, //ny Ky
                                                {0.0}}; //offset
    std::vector<std::vector<double>> baseparamlist = cart_product<double>(paramgen);
    

    std::vector<std::vector<std::vector<double>>> bigpertlist;
    std::vector<double> d_a_vals = {0.01};
    std::vector<double> d_b_vals = {0.05185};
    
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

                                    //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset}
                                    xyz_intyhill system{params[0],params[1],params[2],params[3],
                                                params[4],params[5],params[6],params[7],params[8],
                                                params[9],params[10]};


                                    std::random_device rd;
                                    std::seed_seq sd{rd(), rd(), rd(), rd(),rd()};
                                    rand_eng engine(sd);
                                    
                                    //{lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,offset}
                                    covbal_withpdist alldata{params};
                                    
                                    state_t initialstate = {initial_state_dist(engine),initial_state_dist(engine),initial_state_dist(engine)};
                                    auto begin = std::chrono::high_resolution_clock::now();
                                    long int steps = gsim(system,alldata,initialstate,Nsteps,Nmax,engine);
                                    auto end = std::chrono::high_resolution_clock::now();
                                    auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
                                    double exec_time = static_cast<double>(et.count());
                                    double dsteps = static_cast<double>(steps);
                                    covbaldata_3D data = alldata.cbvals.cbvals;
                                    double w = data.wsum;
                                    double zero = 0;

                                    //saving results {Exec time, sim time, steps, flux balance checks, cov balance checks, used parameters,avgs,variances+covariances,sensitivity_avg,sensitivityatavg}
                            std::vector<double> streaming_data = {exec_time,w,dsteps,
                                                        rel_err(data.meanRpX,data.meanRmX),rel_err(data.meanRpY,data.meanRmY),rel_err(data.meanRpZ,data.meanRmZ),
                                                        cov_balance_comp(data.CXRmX/w,data.CXRmX/w,data.meanRpX+data.meanRmX,data.CXRpX/w,data.CXRpX/w),
                                                        cov_balance_comp(data.CYRmX/w,data.CXRmY/w,0,data.CXRpY/w,data.CYRpX/w),
                                                        cov_balance_comp(data.CYRmY/w,data.CYRmY/w,data.meanRpY+data.meanRmY,data.CYRpY/w,data.CYRpY/w),
                                                        cov_balance_comp(data.CZRmX/w,data.CXRmZ/w,0,data.CZRpX/w,data.CXRpZ/w),
                                                        cov_balance_comp(data.CZRmY/w,data.CYRmZ/w,0,data.CZRpY/w,data.CYRpZ/w),
                                                        cov_balance_comp(data.CZRmZ/w,data.CZRmZ/w,data.meanRpZ+data.meanRmZ,data.CZRpZ/w,data.CZRpZ/w),
                                                        system.lambda,system.beta_x,system.gamma,system.beta_y,system.alpha,system.beta_z,system.nx,system.Kx,
                                                        system.ny,system.Ky,system.offset,
                                                        data.meanx,data.meany,data.meanz,
                                                        data.Vx/w,data.Vy/w,data.Vz/w,data.Cxy/w,data.Cxz/w,data.Cyz/w,
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
                                        streamVector_contline<int>(pdist_file,key,delim);
                                        pdist_file << val << endl;
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