//Library, Data.hpp

/*
 * Library containing structure to store data. All data storage 
 * structures must have a recordData() method that takes a state and 
 * a time spent in that state. Some might need to know dimension
 * (can be retrieved by system) while others (ex running avgs)
 * might need to store information in member variables. This offers 
 * a great deal of flexibility in how you want your information 
 * presented. You can also add output methods for sending data to Python
 * for example
 * 
 * Structures must have a
 * -A data storage mechanism and void recordData function which takes the 
 * 		current state and jumptime. Could be time traces, map of points
 * 		and times, array for prob dist, or combo.
 * 
 * If pre-existing structs have the functionality needed, you can
 * merge them by building a new struct that makes the old ones
 * act as members. Then, the new structs recordData() can call the
 * recordData() of its members.
 * 
 */
#ifndef DATA
#define DATA
#include "Gillespie.hpp"
#include "System.hpp"
#include <unordered_map>
#include <vector>
#include <tuple>
#include <cmath>

constexpr double pi = 3.14159265358979323846;

template <typename T>
struct sized_array{
	int totalelements;
	std::vector<int> dim_sizes;
	std::vector<T> array;
	sized_array(std::vector<int>& dimension_sizes){
		totalelements = std::accumulate(dimension_sizes.begin(), dimension_sizes.end(), 1, std::multiplies<int>());
		dim_sizes = dimension_sizes;
		array.resize(totalelements);
	}
	
};

/*
 * Struct for stable online weighted covariance/variance calculations
 */

struct online_weighted_avg_cov_var{
    double meanx,meany,wsum,wsum2,C,Vx,Vy;
  online_weighted_avg_cov_var(){
    meanx = 0;
    meany = 0;
    wsum = 0; //Total weights
    wsum2 = 0; //Total squared weights
    C = 0; //The covariance
    Vx = 0; //The variance in x
	Vy = 0; //The variance in y
	}
    //one loop of stable calculation. X is first data, Y is second data, w is weight
    void recordData(const state_t &currentstate,const rate_list &rates, double const& w){
        wsum += w;
        wsum2 += w * w;
        double rw = w / wsum;
        double dx = currentstate.at(0) - meanx;
        double dy = currentstate.at(1) - meany;
        meanx += rw * dx;
        meany += rw * dy;
        C += w * dx * dy;
        Vx += w * dx * dx;
        Vy += w * dy * dy;
	}
	//Function to return meanx,meany,varx,vary,covxy
	std::vector<double> genMoments(){
		double covar = C / wsum;
		double varx = Vx / wsum;
		double vary = Vy / wsum;
		//# Bessel's correction for sample variance
		//# Frequency weights
		//sample_frequency_covar = C / (wsum - 1)
		//# Reliability weights
		//sample_reliability_covar = C / (wsum - wsum2 / wsum)
		std::vector<double> moms = {meanx,meany,varx,vary,covar};
		return moms;
	}

};


/*
 * Struct for cov balance check on 2D data
 */

struct covbaldata{
    double meanx,meany,wsum,wsum2,Cxy,Vx,Vy,meanRpX,meanRpY,meanRmX,meanRmY;
    double CXRmX;
    double CXRpX;
    double CYRmX;
    double CYRpX;
    double CXRmY;
    double CXRpY;
    double CYRmY;
    double CYRpY;
  covbaldata(){
    meanx = 0;
    meany = 0;
    wsum = 0; //Total weights
    wsum2 = 0; //Total squared weights
    Cxy = 0; //The covariance
    Vx = 0; //The variance in x
	Vy = 0; //The variance in y
    meanRmX = 0;
    meanRmY = 0;
    meanRpX = 0;
    meanRpY = 0;
    CXRmX = 0;
    CXRpX = 0;
    CYRmX = 0;
    CYRpX = 0;
    CXRmY = 0;
    CXRpY = 0;
    CYRmY = 0;
    CYRpY = 0;
	}
    //one loop of stable calculation. X is first data, Y is second data, w is weight
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        wsum += w;
        double rw = w / wsum;

        double dx = currentstate[0] - meanx;
        double dy = currentstate[1] - meany;

        double dRpX = rates[0] - meanRpX;
        double dRmX = rates[1] - meanRmX;
        double dRpY = rates[2] - meanRpY;
        double dRmY = rates[3] - meanRmY;

        meanx += rw * dx;
        meany += rw * dy;
        meanRpX += rw*dRpX;
        meanRmX += rw*dRmX;
        meanRpY += rw*dRpY;
        meanRmY += rw*dRmY;

        Cxy += w * dx * dy;
        Vx += w * dx * dx;
        Vy += w * dy * dy;

        CXRmX += w*dRmX*dx;
        CXRpX += w*dRpX*dx;

        CYRmX += w*dRmX*dy;
        CYRpX += w*dRpX*dy;
        CXRmY += w*dRmY*dx;
        CXRpY += w*dRpY*dx;
        
        CYRmY += w*dRmY*dy;
        CYRpY += w*dRpY*dy;
	}
	//Function to return meanx,meany,varx,vary,covxy
	std::vector<double> genMoments(){
		double covar = Cxy / wsum;
		double varx = Vx / wsum;
		double vary = Vy / wsum;
		std::vector<double> moms = {meanx,meany,varx,vary,covar};
		return moms;
	}
    std::vector<double> CovBalVals(){
		std::vector<double> cb = {CXRmX/wsum,CXRpX/wsum,
                                 CYRmX/wsum,CYRpX/wsum,
                                 CXRmY/wsum,CXRpY/wsum,
                                 CYRmY/wsum,CYRpY/wsum};
		return cb;
	}

};

/*
 * Struct for cov balance check on 2D data with fb calc
 */

struct covbal_fb{
    covbaldata cbvals;
    std::vector<double> params; //Order {beta_x,gamma,beta_y,lambda_x,nx,Kx,ox,lambda_y,ny,Ky,oy}
    double meanF_y;
    double meanF_x;
    covbal_fb(std::vector<double> pars){
        params = pars;
        cbvals = covbaldata();
        meanF_y = 0.0;
        meanF_x = 0.0;
    }
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        double rw = w / cbvals.wsum;

        int x = currentstate.at(0);
        int y = currentstate.at(1);

        double dFx = hill_loglog_deriv(x,params.at(3),params.at(5),params.at(4),params.at(6))-meanF_x;
        double dFy = hill_loglog_deriv(y,params.at(7),params.at(9),params.at(8),params.at(10))-meanF_y;

        meanF_x += rw * dFx;
        meanF_y += rw * dFy;

    }
};



//From https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
//and then edited to go into a template specialization for std::hash
namespace std {
    template<> struct hash<state_t> {
        std::size_t operator()(state_t const& vec) const {
          std::size_t seed = vec.size();
          for(auto& i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
          }
          return seed;
        }
    };
}


int linear_index(std::vector<int> dimension_sizes,std::vector<int> index){
	int idx = 0;
	for(unsigned int x = 0; x < dimension_sizes.size();x++){
		int temp_idx = index.at(x);
		for(unsigned int y = 0; y < x; y++){
			temp_idx *= dimension_sizes.at(y);
		}
		idx += temp_idx;
	}
	return idx;
}
// //2D unordered map for probability dist
// struct pdist_2D_map{
//     double t_tot;
//     std::unordered_map<state_t, precision> P;
//     void recordData(state_t const& currentstate,rate_list const &rates, precision const& time_jump){
// 		P[currentstate] += time_jump;
// 		t_tot += time_jump;
// 	}
// 	std::unordered_map<state_t,precision> normaliseHist(){
// 		std::unordered_map<state_t,precision> pdist;
// 			for(const auto& x : P){
// 				pdist[x.first] = x.second/t_tot;
// 			}
// 		return pdist;
// 	}
//     std::vector<double> moments(){
//         //meanx,meany,x2,y2,xy
//         std::vector<double> momvec = {0,0,0,0,0};
//         	for(const auto& kv : P){
//                 state_t state = kv.first;
//                 precision time_spent = kv.second;
//                 int x = state.at(0);
//                 int y = state.at(1);
// 				momvec.at(0) += x*time_spent;
//                 momvec.at(1) += y*time_spent;
//                 momvec.at(2) += x*x*time_spent;
//                 momvec.at(3) += y*y*time_spent;
//                 momvec.at(4) += x*y*time_spent;
// 			}
//             for(int i = 0; i < momvec.size();i++){
//                 momvec.at(i) /= t_tot;
//             }
//         return momvec;
//     }
//     sized_array<precision> gen_array(){
// 		std::unordered_map<state_t,precision> pdist = normaliseHist();
// 		std::vector<int> dims((pdist.begin()->first).size(),0);
// 		for(const auto& x : pdist){
// 			for(unsigned int i = 0; i < dims.size();i++){
// 				if(x.first.at(i)+1 > dims.at(i)){
// 					dims.at(i) = x.first.at(i)+1;
// 				}
// 			}
// 		}
// 		sized_array<precision> data_array (dims);
// 		for(const auto& x : pdist){
// 			data_array.array.at(linear_index(dims,x.first)) = x.second;
// 		}
// 		return data_array;	
// 	}
// };


struct vector_map_unordered{
	precision totaltime = 0.0;
	std::unordered_map<state_t,precision> time_hist;
	
	void recordData(state_t const& currentstate,rate_list const & rates, precision const& time_jump){
		time_hist[currentstate] += time_jump;
		totaltime += time_jump;
	}
	
    std::vector<double> moments(){
        //meanx,meany,x2,y2,xy
        std::vector<double> momvec = {0,0,0,0,0};
        	for(const auto& kv : time_hist){
                state_t state = kv.first;
                precision time_spent = kv.second;
                int x = state.at(0);
                int y = state.at(1);
				momvec.at(0) += x*time_spent;
                momvec.at(1) += y*time_spent;
                momvec.at(2) += x*x*time_spent;
                momvec.at(3) += y*y*time_spent;
                momvec.at(4) += x*y*time_spent;
			}
            for(int i = 0; i < momvec.size();i++){
                momvec.at(i) /= totaltime;
            }
        return momvec;
    }

	std::vector<std::pair<state_t,precision> > sortedVector(std::unordered_map<state_t,precision> hist){
		std::vector<std::pair<state_t,precision> > vec(hist.begin(),hist.end());
		sort(vec.begin(),vec.end());
		return vec;
	}
	
	std::unordered_map<state_t,precision> normaliseHist(){
		std::unordered_map<state_t,precision> pdist;
			for(const auto& x : time_hist){
				pdist[x.first] = x.second/totaltime;
			}
		return pdist;
	}
	
	sized_array<precision> gen_array(){
		std::unordered_map<state_t,precision> pdist = normaliseHist();
		std::vector<int> dims((pdist.begin()->first).size(),0);
		for(const auto& x : pdist){
			for(unsigned int i = 0; i < dims.size();i++){
				if(x.first.at(i)+1 > dims.at(i)){
					dims.at(i) = x.first.at(i)+1;
				}
			}
		}
		sized_array<precision> data_array (dims);
		for(const auto& x : pdist){
			data_array.array.at(linear_index(dims,x.first)) = x.second;
		}
		return data_array;	
	}
	
};


std::vector<double> etas(std::vector<double> moms){
    double meanx = moms.at(0);
    double meany = moms.at(1);
    double eta_xx = moms.at(2)/(moms.at(0)*moms.at(0));
    double eta_xy = moms.at(4)/(moms.at(0)*moms.at(1));
    double eta_yy = moms.at(3)/(moms.at(1)*moms.at(1));
    return {meanx,meany,eta_xx,eta_xy,eta_yy};
}

/*
 * Struct for cov balance check on 2D data with shared or. Should do a general version with a function pointer
 */

struct covbal_fb_sharedor{
    covbaldata cbvals;
    std::vector<double> params; //Order {lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,o}
    double meanF_y;
    double meanF_x;
    covbal_fb_sharedor(std::vector<double> pars){
        params = pars;
        cbvals = covbaldata();
        meanF_y = 0.0;
        meanF_x = 0.0;
    }
    covbal_fb_sharedor() = default;
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        double rw = w / cbvals.wsum;

        int x = currentstate.at(0);
        int y = currentstate.at(1);

        //double dFx = ORgate_F_x<int>(x,y,params.at(0),params.at(4),params.at(6),params.at(5),params.at(7),params.at(8))-meanF_x;
        //double dFy = ORgate_F_y<int>(x,y,params.at(0),params.at(4),params.at(6),params.at(5),params.at(7),params.at(8))-meanF_y;

        double fx = F_v(x,params[4],params[5],rates[0]);
        double fy = F_v(y,params[6],params[7],rates[0]);

        double dFx = fx-meanF_x;
        double dFy = fy-meanF_y;

        meanF_x += rw * dFx;
        meanF_y += rw * dFy;
    }

    double F_v(const int& v,const double& nv,const double& Kv,const double& f){
        if(nv==0){return 0.;}
        if(v==0){return 0.;}
        double dv = static_cast<double>(v);
	    return (params.at(0)*nv/f)*((std::pow(dv,nv)/(std::pow(dv,nv)+std::pow(Kv,nv)))*((std::pow(Kv,nv))/(std::pow(dv,nv)+std::pow(Kv,nv))));
    }
};



struct covbal_fb_specifiiedposx_negfy{
    covbaldata cbvals;
    std::vector<double> params; //Order {lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,o}
    double meanF_y;
    double meanF_x;
    double lambda,nx,Kx,ny,Ky;

    covbal_fb_specifiiedposx_negfy(std::vector<double> pars){
        params = pars;
        cbvals = covbaldata();
        meanF_y = 0.0;
        meanF_x = 0.0;
        lambda = params[0];
        nx = params[4];
        Kx = params[5];
        ny = params[6];
        Ky = params[7];
    }
    covbal_fb_specifiiedposx_negfy() = default;
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        double rw = w / cbvals.wsum;

        int x = currentstate[0];
        int y = currentstate[1];
        double f = rates[0];



        double fx = lambda*nx*(std::pow(x,nx)/(std::pow(x,nx)+std::pow(Kx,nx)))*((std::pow(Kx,nx))/(std::pow(x,nx)+std::pow(Kx,nx)))/f;
        double fy = -1*lambda*ny*(std::pow(y,ny)/(std::pow(y,ny)+std::pow(Ky,ny)))*((std::pow(Ky,ny))/(std::pow(y,ny)+std::pow(Ky,ny)))/f;

        double dFx = fx-meanF_x;
        double dFy = fy-meanF_y;

        meanF_x += rw * dFx;
        meanF_y += rw * dFy;
    }
};

struct covbal_fb_sharedor_ratios{
    covbaldata cbvals;
    std::vector<double> params; //Order {lambda,beta_x,gamma,beta_y,nx,Kx,ny,Ky,o}
    double meanF_y;
    double meanF_x;
    double mean_k;
    double mean_atan2k;
    covbal_fb_sharedor_ratios(std::vector<double> pars){
        params = pars;
        cbvals = covbaldata();
        meanF_y = 0.0;
        meanF_x = 0.0;
        mean_k = 0.0;
        mean_atan2k = 0.0;
    }
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        double rw = w / cbvals.wsum;

        int x = currentstate.at(0);
        int y = currentstate.at(1);

        //double dFx = ORgate_F_x<int>(x,y,params.at(0),params.at(4),params.at(6),params.at(5),params.at(7),params.at(8))-meanF_x;
        //double dFy = ORgate_F_y<int>(x,y,params.at(0),params.at(4),params.at(6),params.at(5),params.at(7),params.at(8))-meanF_y;

        double fx = F_v(x,params.at(4),params.at(5),rates.at(0));
        double fy = F_v(y,params.at(6),params.at(7),rates.at(0));
        double k = fy/(1-fx);

        double dFx = fx-meanF_x;
        double dFy = fy-meanF_y;
        double dk = k-mean_k;
        double datan2k = std::atan2(fy,1-fx)-mean_atan2k;

        meanF_x += rw * dFx;
        meanF_y += rw * dFy;
        mean_k += rw*dk;
        mean_atan2k += rw*datan2k;

    }

    double F_v(const int& v,const double& nv,const double& Kv,const double& f){
        if(nv==0){return 0.;}
        if(v==0){return 0.;}
        double dv = static_cast<double>(v);
	    return (params.at(0)*nv/f)*((std::pow(dv,nv)/(std::pow(dv,nv)+std::pow(Kv,nv)))*((std::pow(Kv,nv))/(std::pow(dv,nv)+std::pow(Kv,nv))));
    }
};
struct covbal_fb_yposneg{
    covbaldata cbvals;
    std::vector<double> params; //Order {lambda,beta_x,gamma,beta_y,Kneg,Kpos,o}
    double meanF_y;
    double meanF_x;
    double mean_k;
    double mean_atan2k;
    covbal_fb_yposneg() = default;
    covbal_fb_yposneg(std::vector<double> pars){
        params = pars;
        cbvals = covbaldata();
        meanF_y = 0.0;
        meanF_x = 0.0;
        mean_k = 0.0;
        mean_atan2k = 0.0;
    }
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        double rw = w / cbvals.wsum;

        int x = currentstate.at(0);
        int y = currentstate.at(1);

        double fy = sensitivity(y,rates.at(0));
        double dFy = fy-meanF_y;

        meanF_y += rw * dFy;

    }

    double sensitivity(const int& y,const double& f){
        //Order {lambda 0,beta_x 1,gamma 2,beta_y 3,Kneg 4,Kpos 5,o 6}
        if(y==0){return 0.;}
        double dy = static_cast<double>(y);
        double factor1 = params[4]/((params[4]+dy)*(params[5]+dy));
        double factor2 = ((params[5]/(params[5]+dy))-(dy/(params[4]+dy)));
	    return (params[0]*y/f)*factor1*factor2;
    }
};


/*
 * Struct for cov balance check on 3D data
 */

struct covbaldata_3D{
    double meanx=0,meany=0,meanz=0,wsum=0,wsum2=0;
    double Cxy=0,Cxz=0,Cyz=0,Vx=0,Vy=0,Vz=0;
    double meanRpX=0,meanRpY=0,meanRmX=0,meanRmY=0,meanRpZ=0,meanRmZ=0;
    double CXRmX=0,CXRpX=0,CYRmX=0,CYRpX=0,CZRmX=0,CZRpX=0;
    double CXRmY=0,CXRpY=0,CYRmY=0,CYRpY=0,CZRmY=0,CZRpY=0;
    double CXRmZ=0,CXRpZ=0,CYRmZ=0,CYRpZ=0,CZRmZ=0,CZRpZ=0;
//one loop of stable calculation. X is first data, Y is second data, w is weight
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        wsum += w;
        double rw = w / wsum;

        double dx = currentstate[0] - meanx;
        double dy = currentstate[1] - meany;
        double dz = currentstate[2] - meanz;

        double dRpX = rates[0] - meanRpX;
        double dRmX = rates[1] - meanRmX;
        double dRpY = rates[2] - meanRpY;
        double dRmY = rates[3] - meanRmY;
        double dRpZ = rates[4] - meanRpZ;
        double dRmZ = rates[5] - meanRmZ;

        meanx += rw * dx;
        meany += rw * dy;
        meanz += rw * dz;
        meanRpX += rw*dRpX;
        meanRmX += rw*dRmX;
        meanRpY += rw*dRpY;
        meanRmY += rw*dRmY;
        meanRpZ += rw*dRpZ;
        meanRmZ += rw*dRmZ;

        Cxy += w * dx * dy;
        Cxz += w * dx * dz;
        Cyz += w * dy * dz;
        Vx += w * dx * dx;
        Vy += w * dy * dy;
        Vz += w * dz * dz;

        CXRmX += w*dRmX*dx;
        CXRpX += w*dRpX*dx;
        CYRmX += w*dRmX*dy;
        CYRpX += w*dRpX*dy;
        CZRmX += w*dRmX*dz;
        CZRpX += w*dRpX*dz;

        CXRmY += w*dRmY*dx;
        CXRpY += w*dRpY*dx;
        CYRmY += w*dRmY*dy;
        CYRpY += w*dRpY*dy;
        CZRmY += w*dRmY*dz;
        CZRpY += w*dRpY*dz;

        CXRmZ += w*dRmZ*dx;
        CXRpZ += w*dRpZ*dx;
        CYRmZ += w*dRmZ*dy;
        CYRpZ += w*dRpZ*dy;
        CZRmZ += w*dRmZ*dz;
        CZRpZ += w*dRpZ*dz;
	}
	//Function to return meanx,meany,varx,vary,covxy
	std::vector<double> genMoments(){
		double covar = Cxy / wsum;
		double varx = Vx / wsum;
		double vary = Vy / wsum;
		std::vector<double> moms = {meanx,meany,varx,vary,covar};
		return moms;
	}
    std::vector<double> CovBalVals(){
		std::vector<double> cb = {CXRmX/wsum,CXRpX/wsum,
                                 CYRmX/wsum,CYRpX/wsum,
                                 CXRmY/wsum,CXRpY/wsum,
                                 CYRmY/wsum,CYRpY/wsum};
		return cb;
	}

};

struct covbal_inthill{
    covbaldata_3D cbvals;
    std::vector<double> params; //Order {lambda,beta_x,gamma,beta_y,alpha,beta_z,nx,Kx,ny,Ky,offset}
    //Sensitivity of x to y: sensitivity of z to y. Similarly sensitivity of x to x is sensitivity of z to x.
    double meanF_y;
    double meanF_x;
    double lambda,nx,Kx,ny,Ky,alpha;
    covbal_inthill() = default;
    covbal_inthill(std::vector<double> pars){
        params = pars;
        cbvals = covbaldata_3D();
        meanF_y = 0.0;
        meanF_x = 0.0;
        lambda = params[0];
        alpha = params[4];
        nx = params[6];
        Kx = params[7];
        ny = params[8];
        Ky = params[9];
    }
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        double rw = w / cbvals.wsum;
        int x = currentstate[0];
        int y = currentstate[1];
        int z = currentstate[2];
        double Rzp = rates[4];
        //fx is equivalent to dlog(Rz+)/dlog(x)
        //fy is equivalent to dlog(Rz+)/dlog(y)
        double fx = nx*(1/Rzp)*hillderiv(x,nx,Kx);
        double fy = ny*alpha*(1/Rzp)*hillderiv(y,ny,Ky);

        double dFx = fx-meanF_x;
        double dFy = fy-meanF_y;

        meanF_x += rw * dFx;
        meanF_y += rw * dFy;

    }
    double hillderiv(const int& v,const double& nv,const double& Kv){
        if(nv==0){return 0.;}
        if(v==0){return 0.;}
        double dv = static_cast<double>(v);
	    return ((std::pow(dv,nv)/(std::pow(dv,nv)+std::pow(Kv,nv)))*((std::pow(Kv,nv))/(std::pow(dv,nv)+std::pow(Kv,nv))));
    }
};


struct covbal_weirdfb{
    covbaldata cbvals;
    std::vector<double> params; //Order {lambda,beta_x,gamma,beta_y,nx,K,offset}
    double meanF_y;
    double meanF_x;
    covbal_weirdfb(std::vector<double> pars){
        params = pars;
        cbvals = covbaldata();
    }
    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        cbvals.recordData(currentstate,rates,w);
        double rw = w / cbvals.wsum;

        double fx = params[-3]*(1-params[-1]/rates[0]);
        double fy = -1*(currentstate[1]/(params[4]+currentstate[1])*(1-params[-1]/rates[0]));
        
        double dFx = fx-meanF_x;
        double dFy = fy-meanF_y;

        meanF_x += rw * dFx;
        meanF_y += rw * dFy;
    }
};

struct covbaldata_4D{
    double meanx=0,meany=0,meanz1=0,meanz2=0,wsum=0,wsum2=0;
    double Cxy=0,Cxz1=0,Cxz2=0,Cyz1=0,Cyz2=0,Cz1z2=0,Vx=0,Vy=0,Vz1=0,Vz2=0;
    double meanRpX=0,meanRpY=0,meanRmX=0,meanRmY=0,meanRpZ1=0,meanRmZ1=0,meanRpZ2=0,meanRmZ2=0;
    double CXRmX=0,CXRpX=0,CYRmX=0,CYRpX=0,CZ1RmX=0,CZ1RpX=0,CZ2RmX=0,CZ2RpX=0;
    double CXRmY=0,CXRpY=0,CYRmY=0,CYRpY=0,CZ1RmY=0,CZ1RpY=0,CZ2RmY=0,CZ2RpY=0;
    double CXRmZ1=0,CXRpZ1=0,CYRmZ1=0,CYRpZ1=0,CZ1RmZ1=0,CZ1RpZ1=0,CZ2RmZ1=0,CZ2RpZ1=0;
    double CXRmZ2=0,CXRpZ2=0,CYRmZ2=0,CYRpZ2=0,CZ1RmZ2=0,CZ1RpZ2=0,CZ2RmZ2=0,CZ2RpZ2=0;

    void recordData(const state_t &currentstate,const rate_list &rates,const double &w){
        wsum += w;
        double rw = w / wsum;
        double dx = currentstate[0] - meanx;
        double dy = currentstate[1] - meany;
        double dz1 = currentstate[2] - meanz1;
        double dz2 = currentstate[3] - meanz2;

        double dRpX = rates[0] - meanRpX;
        double dRmX = rates[1] - meanRmX;
        double dRpY = rates[2] - meanRpY;
        double dRmY = rates[3] - meanRmY;
        double dRpZ1 = rates[4] - meanRpZ1;
        double dRmZ1 = rates[5] - meanRmZ1;
        double dRpZ2 = rates[6] - meanRpZ2;
        double dRmZ2 = rates[7] - meanRmZ2;

        meanx += rw * dx;
        meany += rw * dy;
        meanz1 += rw * dz1;
        meanz2 += rw * dz2;
        meanRpX += rw*dRpX;
        meanRmX += rw*dRmX;
        meanRpY += rw*dRpY;
        meanRmY += rw*dRmY;
        meanRpZ1 += rw*dRpZ1;
        meanRmZ1 += rw*dRmZ1;
        meanRpZ2 += rw*dRpZ2;
        meanRmZ2 += rw*dRmZ2;

        Cxy += w * dx * dy;
        Cxz1 += w * dx * dz1;
        Cxz2 += w * dx * dz2;
        Cyz1 += w * dy * dz1;
        Cyz2 += w * dy * dz2;
        Cz1z2 += w * dz1 * dz2;
        Vx += w * dx * dx;
        Vy += w * dy * dy;
        Vz1 += w * dz1 * dz1;
        Vz2 += w * dz2 * dz2;

        CXRmX += w*dRmX*dx;
        CXRpX += w*dRpX*dx;
        CYRmX += w*dRmX*dy;
        CYRpX += w*dRpX*dy;
        CZ1RmX += w*dRmX*dz1;
        CZ1RpX += w*dRpX*dz1;
        CZ2RmX += w*dRmX*dz2;
        CZ2RpX += w*dRpX*dz2;

        CXRmY += w*dRmY*dx;
        CXRpY += w*dRpY*dx;
        CYRmY += w*dRmY*dy;
        CYRpY += w*dRpY*dy;
        CZ1RmY += w*dRmY*dz1;
        CZ1RpY += w*dRpY*dz1;
        CZ2RmY += w*dRmY*dz2;
        CZ2RpY += w*dRpY*dz2;

        CXRmZ1 += w*dRmZ1*dx;
        CXRpZ1 += w*dRpZ1*dx;
        CYRmZ1 += w*dRmZ1*dy;
        CYRpZ1 += w*dRpZ1*dy;
        CZ1RmZ1 += w*dRmZ1*dz1;
        CZ1RpZ1 += w*dRpZ1*dz1;
        CZ2RmZ1 += w*dRmZ1*dz2;
        CZ2RpZ1 += w*dRpZ1*dz2;

        CXRmZ2 += w*dRmZ2*dx;
        CXRpZ2 += w*dRpZ2*dx;
        CYRmZ2 += w*dRmZ2*dy;
        CYRpZ2 += w*dRpZ2*dy;
        CZ1RmZ2 += w*dRmZ2*dz1;
        CZ1RpZ2 += w*dRpZ2*dz1;
        CZ2RmZ2 += w*dRmZ2*dz2;
        CZ2RpZ2 += w*dRpZ2*dz2;
    }

    std::vector<double> genMoments(){
        double covarxy = Cxy / wsum;
        double covarxz1 = Cxz1 / wsum;
        double covarxz2 = Cxz2 / wsum;
        double covaryz1 = Cyz1 / wsum;
        double covaryz2 = Cyz2 / wsum;
        double covarz1z2 = Cz1z2 / wsum;
        double varx = Vx / wsum;
        double vary = Vy / wsum;
        double varz1 = Vz1 / wsum;
        double varz2 = Vz2 / wsum;
        std::vector<double> moms = {meanx,meany,meanz1,meanz2,varx,vary,varz1,varz2,covarxy,covarxz1,covarxz2,covaryz1,covaryz2,covarz1z2};
        return moms;
    }

    std::vector<double> CovBalVals(){
        std::vector<double> cb = {CXRmX/wsum,CXRpX/wsum,
                                 CYRmX/wsum,CYRpX/wsum,
                                 CZ1RmX/wsum,CZ1RpX/wsum,
                                 CZ2RmX/wsum,CZ2RpX/wsum,
                                 CXRmY/wsum,CXRpY/wsum,
                                 CYRmY/wsum,CYRpY/wsum,
                                 CZ1RmY/wsum,CZ1RpY/wsum,
                                 CZ2RmY/wsum,CZ2RpY/wsum,
                                 CXRmZ1/wsum,CXRpZ1/wsum,
                                 CYRmZ1/wsum,CYRpZ1/wsum,
                                 CZ1RmZ1/wsum,CZ1RpZ1/wsum,
                                 CZ2RmZ1/wsum,CZ2RpZ1/wsum,
                                 CXRmZ2/wsum,CXRpZ2/wsum,
                                 CYRmZ2/wsum,CYRpZ2/wsum,
                                 CZ1RmZ2/wsum,CZ1RpZ2/wsum,
                                 CZ2RmZ2/wsum,CZ2RpZ2/wsum};
        return cb;
    }


};
#endif
