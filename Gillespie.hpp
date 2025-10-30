//LIBRARY, gsim.hpp

#ifndef GILLESPIE
#define GILLESPIE

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <numeric>
#include <chrono>
#include <tuple>

using precision = double; //use to control precision of time values
using rand_eng = std::mt19937; //use to control random engine
using state_t = std::vector<int>;
using rate_list = std::vector<precision>;

/*
 * The first set of templates and functions are useful utilities
 * used throughout my own research, which I've enclosed here for
 * anyone who might also find them useful
 */

double rel_err(double x,double y){
    return std::abs(y-x)/std::abs((y+x)/2.0);
}

std::vector<double> rel_error_vecs(std::vector<double> expected,std::vector<double> etas){
    std::vector<double> relerr(expected.size());
    for(int i=0;i<expected.size();i++){
        relerr.at(i) = std::abs(expected.at(i)-etas.at(i))/expected.at(i);
    }
    return relerr;
}

//Following template for vector streaming
template <typename T>
std::ostream& operator<< (std::ostream &o, std::vector<T> const& v) {  
	o << "[";  
	auto delim = "";  
	for (auto const& i : v) {    
		o << delim << i;    
		delim = ", ";  
		}  
	return o << "]";
	}
//Pair Streaming
template <typename T,typename T2>
std::ostream& operator<< (std::ostream &o, std::pair<T,T2> const& v) {  
	o << "(";  
	o << v.first << "," << v.second;
	return o << ")";
	}

std::string return_current_time_and_date()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%m-%d %H%M");
    return ss.str();
}

int gen_seed(){
	//return 10; //this lets you fix a seed for the whole program (deterministic)
	//Clock based seeding. This is okay.
	return static_cast<int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	//To do: add a check on std::random_device().entropy. If its 0, use clock based. Otherwise use random device.
}

/*Relative Error Calculation
 */

double rel_error(double RHS, double LHS){
	double val;
	if(RHS == 0){
		if(LHS == 0){
			val = 0;
		}
		else{
			val = abs(LHS);
		}
	}
	else if(LHS==0){val = abs(RHS);}
	else{val = abs(RHS - LHS)/abs((RHS+LHS)/2);}
	return val;
}


/*
 * Kahan Sum for stable average calculations
 */
 
void KahanSum(precision& current_total,precision& compensator,precision added_val){
	precision y = added_val - compensator;
	precision t = current_total + y;
	compensator = (t-current_total) - y;
	current_total = t;
}


/*
 * The following functions and templates are the actual Gillespie 
 * simulation utilities.
 */

//given list of rates find jumptime
precision findJumptime(precision const& r_tot,rand_eng& engine){
	std::exponential_distribution<precision> distribution (r_tot);
	precision t_jump = distribution(engine);
	return t_jump;
}

//Jumptime function that doesn't need exponential dist
precision findJumptime_uniform(precision const& r_tot, rand_eng& engine,std::uniform_real_distribution<precision> distribution){
	precision rand_uni = distribution(engine);
	precision t_jump = -1.0*std::log(1 - rand_uni) / r_tot;
	return t_jump;
}

unsigned int chooseProcess(rate_list const& rates, precision const& r_tot, rand_eng& engine,std::uniform_real_distribution<precision> distribution){
	precision p = distribution(engine);
	precision pointed_val = p*r_tot;
	precision ratesum = 0.0;
	for(unsigned int proc = 0; proc < rates.size()-1;proc++){
		ratesum += rates[proc]; //Keeps running sum of rates up until now
		if(pointed_val <= ratesum){
			return proc;
		}
	}
	//Triggers if you didn't find the value in the partial summation. 
	//Needs error check in case you get p*r_tot > r_tot
	return rates.size()-1; 
	
}
double cov_balance_comp(double CiRmj,double CjRmi,double Dij,double CiRpj,double CjRpi){
    double LHS = CiRmj+CjRmi;
    double RHS = CiRpj+CjRpi+Dij;
    return std::abs(RHS-LHS)/std::abs((RHS+LHS)/2.0);
}


//


 /*
 * Gillespie Simulation Driver function. 
 * type system needs to have member functions getRates and getStates,
 * appropriately formatted. Examples can be found in Systems.hpp.
 * 
 */
template<typename system,typename datastore>
long int gsim(system& s,datastore& d, state_t& initialstate,long int Nsteps,long int Nmax,rand_eng& engine){

	state_t currentstate = initialstate;
	rate_list rates(s.Numprocs);
	std::vector<int> counts(s.Numprocs,Nsteps);
	long int steps = 0; //Check for maximum number of steps
	long int globalcount = Nsteps*s.Numprocs;
	
	std::uniform_real_distribution<precision> uni_distribution(0.0,1.0);

	while( globalcount > 0 && steps < Nmax){
		//loop: while each process has run less than Nsteps times, or system hasnt reached Nmx steps
		s.updateRates(currentstate,rates);

		//Get total rate
		precision r_tot = std::accumulate(rates.begin(),rates.end(),0.0);
		if(r_tot <= 0.0){ //change to within an epsilon
			std::cout << "No processes are going to occur. \n";
			std::cout << "The absorbing state is " << currentstate <<"\n";
			std::cout << "The remaining rates are " <<rates << "\n";
			std::cout << "This had a total rate of " <<r_tot <<"\n";
			std::cout << "Remaining Steps " << counts <<"\n";
			std::cout << "Taken Steps " << steps <<"\n";
			break;
		}
        
		//Get time in state before jump
		precision t_jump = findJumptime_uniform(r_tot,engine,uni_distribution);
		//Record time spent in current state
		d.recordData(currentstate,rates,t_jump);
		//Choose the process that occured and update state
		int proc = chooseProcess(rates,r_tot,engine,uni_distribution);
		s.updateState(proc,currentstate);
		//Add step, and remove count as appropriate
		steps++;
		if(counts[proc] > 0){
			globalcount -= 1;
			counts[proc] -= 1;
		}
	}
	return steps;
}
#endif