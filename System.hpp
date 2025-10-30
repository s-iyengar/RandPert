#ifndef SYSTEM
#define SYSTEM

#include "Gillespie.hpp"
#include <cmath>

template<typename numtype>
double unscaled_hf(const numtype& x,double& K, double& n){
	//Def: f(x) = (x^n)/(x^n+K^n)
    if(n == 0){
		if(K == 0){return 0;}
		else{return 0.5;}
	}
	if(x == 0){
		if(n < 0){return 0.5;}
		else{return 0;}
	}
    else{
		double fx = static_cast<double>(x);
        return std::pow(fx,n)/(std::pow(fx,n) + std::pow(K,n));
    }
};

template<typename numtype>
double unscaled_hf_deriv(const numtype& x,double& K, double& n){
	//Def: f(x) = (x^n)/(x^n+K^n)
    if(n == 0){return 0;}
	if(x == 0){return 0;}
    else{
		double fx = static_cast<double>(x);
        return n*(std::pow(fx,n-1)/(std::pow(fx,n) + std::pow(K,n)))*(std::pow(K,n)/(std::pow(fx,n) + std::pow(K,n)));
    }
};

template<typename numtype>
double ORgate_hf(const numtype& x,const numtype& y,double&lambda,double&nx,double&ny,double&Kx,double&Ky,double&offset){
	return lambda*(unscaled_hf<numtype>(x,Kx,nx)+unscaled_hf<numtype>(y,Ky,ny)+offset);
}

template<typename numtype>
double ORgate_F_x(const numtype& x,const numtype& y,double&lambda,double&nx,double&ny,double&Kx,double&Ky,double&offset){
	if(nx==0){return 0.;}
	if(x==0){return 0.;}
	return (lambda*x)*(unscaled_hf_deriv<numtype>(x,Kx,nx)/ORgate_hf<numtype>(x,y,lambda,nx,ny,Kx,Ky,offset));
}

template<typename numtype>
double ORgate_F_y(const numtype& x,const numtype& y,double&lambda,double&nx,double&ny,double&Kx,double&Ky,double&offset){
	if(ny==0){return 0.;}
	if(y==0){return 0.;}
	return (lambda*y)*(unscaled_hf_deriv<numtype>(y,Ky,ny)/ORgate_hf<numtype>(x,y,lambda,nx,ny,Kx,Ky,offset));
}


double hill_function(int& x,double& lambda,double& K, double& n,double & offset){
	//Def: f(x) = lambda*((x^n)/(x^n+K^n) + offset)
    if(n == 0){return lambda*(0.5+offset);}
	if(x == 0){
		if(n < 0){return lambda*(0.5+offset);}
		else{return lambda*offset;}
	}
    else{
		float fx = static_cast<float>(x);
        return lambda*(std::pow(fx,n)/(std::pow(fx,n) + std::pow(K,n))+offset);
    }
};

double hill_function_pure(int& x,double& K, double& n){
	//Def: f(x) = lambda*((x^n)/(x^n+K^n) + offset)
    if(n == 0){return 0.;
	}
	if(x == 0){
		if(n < 0){return 1.;}
		else{return 0;}
	}
    else{
		double fx = static_cast<double>(x);
        return std::pow(fx,n)/(std::pow(fx,n) + std::pow(K,n));
    }
};

/* double hill_function(int& x,float& lambda,float& K, float& n,float & offset){
    if(n == 0){return lambda+offset;}
    else if(n < 0){
        
        if(x == 0){return lambda+offset;}
        else{
            float fx = static_cast<float>(x);
            return lambda/(std::pow(fx/K,n) + 1)+offset;
        }
    }
    else{
        if(x == 0){return offset;}
        else{
            float fx = static_cast<float>(x);
            return lambda*std::pow(fx,n)/(std::pow(fx,n) + std::pow(K,n))+offset;
            }
    }
}; */

double hill_function_double(double& x,double& lambda,double& K, double& n,double & offset){
	//Def: f(x) = lambda*((x^n)/(x^n+K^n) + offset)
    if(n == 0){return lambda*(0.5+offset);}
	if(x == 0){
		if(n < 0){return lambda*(0.5+offset);}
		else{return lambda*offset;}
	}
    else{
        return lambda*(std::pow(x,n)/(std::pow(x,n) + std::pow(K,n))+offset);
    }
};

double hill_loglog_deriv(int& x,double& lambda,double& K, double& n,double & offset){
    if(n==0){return 0;}
	if(x==0){return 0;}
    double h = hill_function(x,lambda,K,n,offset);
	double denom = std::pow(x,n) + std::pow(K,n);
    return lambda*n*(std::pow(x,n)/denom)*(std::pow(K,n)/denom)/h;
}

double hill_loglog_deriv_double(double& x,double& lambda,double& K, double& n,double & offset){
    if(n==0){return 0;}
	if(x==0){return 0;}
    double h = hill_function_double(x,lambda,K,n,offset);
	double denom = std::pow(x,n) + std::pow(K,n);
    return lambda*n*(std::pow(x,n)/denom)*(std::pow(K,n)/denom)/h;
}

struct xy_noFB{
	 //Members: four constants chosen at initialisation, two
	 //constants of the simulation
	 double lambda;
     double beta_x;
     double gamma;
     double beta_y;
	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = lambda;
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }
};

struct xy_yFB{
	 //Members: four constants chosen at initialisation, two
	 //constants of the simulation
	 double lambda;
     double beta_x;
     double gamma;
     double beta_y;
	 double n;
	 double K;
     double offset;
	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	double xbirth(int y){
		return lambda*(unscaled_hf(y,K,n)+offset);
	}

	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = xbirth(y);
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }
};


struct xy_xFB{
	 //Members: four constants chosen at initialisation, two
	 //constants of the simulation
	double lambda;
    double beta_x;
    double gamma;
    double beta_y;
	double n;
	double K;
    double offset;
	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = hill_function(x,lambda,K,n,offset);
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }
};

struct xy_xORyFB{
	 //Members: four constants chosen at initialisation, two
	 //constants of the simulation
    double beta_x;
    double gamma;
    double beta_y;

    double lambda_x;
	double nx;
	double Kx;
    double offsetx;

    double lambda_y;
    double ny;
	double Ky;
    double offsety;
	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = hill_function(x,lambda_x,Kx,nx,offsetx)+hill_function(y,lambda_y,Ky,ny,offsety);
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }
};

struct xy_xORyFB_shared{
	 //OR gate with shared lambda and offset
     double lambda;
	 double beta_x;
     double gamma;
     double beta_y;
	 
	 double nx;
	 double Kx;
     double ny;
	 double Ky;
	 double offset;

	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state[0];
		 int y = state[1];
		 //calculate rates
		 rates.at(0) = lambda*(hill_function_pure(x,Kx,nx)+hill_function_pure(y,Ky,ny)+offset);
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate[0] += 1;
				break;
			case 1: //mRNA death
				currentstate[0] -= 1;
				break;
			case 2: //protein birth
				currentstate[1] += 1;
				break;
			case 3: //protein death
				currentstate[1] -= 1;
				break;
		}
	 }

};


struct xy_yposandneg{
	 //Feedback loop where 
	 //R_X^+ = lambda*((kneg/kneg+y)*(y/y+cpos)+offset) linear rates otherwise
     double lambda;
	 double beta_x;
     double gamma;
     double beta_y;
	 
	 double Kneg;
	 double Kpos;
	 double offset;

	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = lambda*(((Kneg)/((double)y+Kneg))*(((double)y)/((double)y+Kpos)))+offset;
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }

};


struct xyz_intneghill{
	 //system where Rx+ = lambda*(K/K+z) all other rates linear
     double lambda;
	 double beta_x;
     double gamma;
     double beta_y;
	 double alpha; //z production rate
	 double beta_z; // z degredation rate
	 double K;

	 static int const dimensions = 3;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 6;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state[0];
		 int y = state[1];
		 int z = state[2];
		 //calculate rates
		 rates[0] = lambda*(K/(K+z));
		 rates[1] = beta_x*x;
		 rates[2] = gamma*x;
		 rates[3] = beta_y*y;
		 rates[4] = alpha*y;
		 rates[5] = beta_z*z;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //x birth
				currentstate.at(0) += 1;
				break;
			case 1: //x death
				currentstate.at(0) -= 1;
				break;
			case 2: //y birth
				currentstate.at(1) += 1;
				break;
			case 3: //y death
				currentstate.at(1) -= 1;
				break;
			case 4: //z birth
				currentstate.at(2) += 1;
				break;
			case 5: //z death
				currentstate.at(2) -= 1;
				break;
		}
	 }

};

struct xy_xORyFB_nooffsetperts{
	 //OR gate with shared lambda and offset
     double lambda;
	 double beta_x;
     double gamma;
     double beta_y;
	 
	 double nx;
	 double Kx;
     double ny;
	 double Ky;
	 double offset;

	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = lambda*(hill_function_pure(x,Kx,nx)+hill_function_pure(y,Ky,ny))+offset;
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }

};



struct xyz_intyhill{
	 //system where Rx+ = alpha*z, Rz+ is hill function all other rates linear
     double lambda;
	 double beta_x;
     double gamma;
     double beta_y;
	 double alpha; //z production rate
	 double beta_z; // z degredation rate
	 double nx;
	 double Kx;
     double ny;
	 double Ky;
	 double offset;

	 static int const dimensions = 3;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 6;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state[0];
		 int y = state[1];
		 int z = state[2];
		 //calculate rates
		 rates[0] = lambda*z;
		 rates[1] = beta_x*x;
		 rates[2] = gamma*x;
		 rates[3] = beta_y*y;
		 rates[4] = alpha*hill_function_pure(y,Ky,ny)+hill_function_pure(x,Kx,nx)+offset;
		 rates[5] = beta_z*z;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //x birth
				currentstate[0] += 1;
				break;
			case 1: //x death
				currentstate[0] -= 1;
				break;
			case 2: //y birth
				currentstate[1] += 1;
				break;
			case 3: //y death
				currentstate[1] -= 1;
				break;
			case 4: //z birth
				currentstate[2] += 1;
				break;
			case 5: //z death
				currentstate[2] -= 1;
				break;
		}
	 }

};


struct xy_posfx_or_negfy{
	 //OR gate with shared lambda and offset. Soecify hill coefficients as positive numbers
     double lambda;
	 double beta_x;
     double gamma;
     double beta_y;
	 
	 double nx;
	 double Kx;
     double ny;
	 double Ky;

	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = lambda*(std::pow(x,nx)/(std::pow(x,nx)+std::pow(Kx,nx)) + std::pow(Ky,ny)/(std::pow(y,ny)+std::pow(Ky,ny)));
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }

};

struct weird_bothfb_xy{
	 //Rx+ = lambda*(x^2*(K/K+y))+offset
     double lambda;
	 double beta_x;
     double gamma;
     double beta_y;
	 
	 double K;
	 double nx;
	 double offset;

	 static int const dimensions = 2;
	 //Numprocs must match the length of rates output
	 static int const Numprocs = 4;
	 
	 //Given a state, calculate the four rates.
	 void updateRates(state_t const &state,rate_list &rates){
		 //retrieve state vars
		 int x = state.at(0);
		 int y = state.at(1);
		 //calculate rates
		 rates.at(0) = lambda*(std::pow(x,nx) * K/(K+y))+offset;
		 rates.at(1) = beta_x*x;
		 rates.at(2) = gamma*x;
		 rates.at(3) = beta_y*y;
	 }
	 
	 //Given an integer correspending to an index of the outputs of
	 //getRates() and the current state, yield the new state.
	 void updateState(const int& proc,state_t &currentstate){
		 switch(proc){
			case 0: //mRNA birth
				currentstate.at(0) += 1;
				break;
			case 1: //mRNA death
				currentstate.at(0) -= 1;
				break;
			case 2: //protein birth
				currentstate.at(1) += 1;
				break;
			case 3: //protein death
				currentstate.at(1) -= 1;
				break;
		}
	 }

};

#endif