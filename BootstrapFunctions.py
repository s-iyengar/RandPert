import numpy as np
import itertools

def lin_eta_xx(eta_ll,eta_gg,fy,fx):
    return (eta_ll + eta_gg*fy**2)/(1-fy-fx)**2
def lin_eta_yy(eta_ll,eta_gg,fy,fx):
    return (eta_ll + eta_gg*(1-fx)**2)/(1-fy-fx)**2
def lin_eta_xy(eta_ll,eta_gg,fy,fx):
    return (eta_ll + eta_gg*(1-fx)*fy)/(1-fy-fx)**2
def lin_rho(eta_ll,eta_gg,fy,fx):
    return lin_eta_xy(eta_ll,eta_gg,fy,fx)/np.sqrt(lin_eta_xx(eta_ll,eta_gg,fy,fx)*lin_eta_yy(eta_ll,eta_gg,fy,fx))
def lin_cvratio(eta_ll,eta_gg,fy,fx):
    return np.sqrt(lin_eta_xx(eta_ll,eta_gg,fy,fx)/lin_eta_yy(eta_ll,eta_gg,fy,fx))

def einsum_corr(x,y):
    """
    Using np.einsum, calculate the n correlation coefficients for two sets of vectors of (n,s)
    """
    (n,s) = x.shape #n samples [ex.num samples of bootstrap] for s datapoints
    mx = np.einsum(x,[0,1],[0])/np.float64(s)
    my = np.einsum(y,[0,1],[0])/np.float64(s)
    dx = (x.T - mx).T #transposing required to get numpy to elementwise subtract automatically
    dy = (y.T - my).T 
    cxy = np.einsum(dx,[0,1],dy,[0,1],[0]) #no scaling since will cancel anyway
    vx = np.einsum(dx,[0,1],dx,[0,1],[0])
    vy = np.einsum(dy,[0,1],dy,[0,1],[0])
    return cxy/np.sqrt(vx*vy)

def einsum_mean_cov_vars(x,y,correction=0):
    """
    Using np.einsum, calculate, for two sets of vectors of (n,s) their means,variances, and covariances
     in each of the n samples (ex from bootstrap) across the s datapoints. Use correction to change scale
    on value (ex set to 1 for bessel)
    """
    (n,s) = x.shape #n samples [ex.num samples of bootstrap] for s datapoints
    mx = np.einsum(x,[0,1],[0])/s
    my = np.einsum(y,[0,1],[0])/s
    dx = (x.T - mx).T #transposing required to get numpy to elementwise subtract automatically
    dy = (y.T - my).T 
    cxy = np.einsum(dx,[0,1],dy,[0,1],[0])/(s-correction)
    vx = np.einsum(dx,[0,1],dx,[0,1],[0])/(s-correction)
    vy = np.einsum(dy,[0,1],dy,[0,1],[0])/(s-correction)
    return mx,my,vx,vy,cxy

def cov_single(x,y,correction=0):
    """
    Calculate covariances, calculating the means
    """
    s = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    dx = x-mx
    dy = y-my
    cxy = np.dot(dx,dy)/(s-correction)
    return cxy

def cov_precalcdmeans(x,y,mx,my,correction=0):
    if len(x.shape) == 1:
        #just do dot product
        s = len(x)
        dx = x-mx #transposing required to get numpy to elementwise subtract automatically
        dy = y-my
        return np.dot(dx,dy)/(s-correction)
    else:
        (n,s) = x.shape[-1]
        dx = (x.T - mx).T #transposing required to get numpy to elementwise subtract automatically
        dy = (y.T - my).T 
        return np.einsum(dx,[0,1],dy,[0,1],[0])/(s-correction)

def bootstrap_gen(data,n_boots):
    """
    Take loaded data from a perturbation data csv and create a bootstrap sample of it.
    Drawn from https://stackoverflow.com/a/46916804
    """
    return data[(np.random.rand(n_boots,len(data))*len(data)).astype(int)]

def replicate_bootstrapping(data,n_boots,n_reps=3,n_perts=9):
    """
    Like bootstrap gen, but pick a single index from every n_replicates rows
    """
    return np.swapaxes(data[(np.mgrid[0:n_perts][:,None]*n_reps)+np.random.choice(n_reps,size=(n_perts,n_boots))],0,1)

def full_bootstrap(data,n_reps=3,n_perts=9):
    """
    Like replicate bootstrapping, but get all possible combinations
    """
    return data[(np.mgrid[0:n_perts][None,:]*n_reps)+np.array(list(itertools.product(list(range(n_reps)),repeat=n_perts)))]

def pertexp_interpret(dataframe,n_boots='all',
                           analysed_columns=['<x>','<y>',"<F_x>","<F_y>"],
                           paramlist=['lambda','gamma'],
                           low_p=2.5,high_p=97.5,nreps=3,nperts=9):
    """
    
    """
    analysed_data_dict = {}
    parammeans = dataframe[paramlist].mean()
    paramvars = dataframe[paramlist].var()
    avgfy,avgfx = dataframe['<F_y>'].mean(),dataframe['<F_x>'].mean()
    sampleell,samplegg = paramvars['lambda']/parammeans['lambda']**2,paramvars['gamma']/parammeans['gamma']**2

    parammeannames = (f"E[{v}]" for v in paramlist)
    paramvarnames = (f"Var[{v}]" for v in paramlist)
    parammeanres = zip(parammeannames,parammeans)
    paramvarres = zip(paramvarnames,paramvars)

    data_to_bs = dataframe[analysed_columns].to_numpy()
    #Generate bootstrap distribution of molecule and fb data values
    #bootstrapped_data = replicate_bootstrapping(data_to_bs,n_boots,n_reps=nreps,n_perts=nperts)
    #bootstrapped_data = bootstrap_gen(data_to_bs,n_boots)
    if n_boots == 'all':
        bootstrapped_data = full_bootstrap(data_to_bs)
    else:
        bootstrapped_data = replicate_bootstrapping(data_to_bs,n_boots,n_reps=nreps,n_perts=nperts)

    #Bootstrap to get means and 95%CI for the 
    #<x>,<y>,avgs of feedbacks,eta_xx/eta_yy,rho_xy,eta_xx,eta_yy,eta_xy,
    #eta_xx+eta_yy,(etaxy-etaxx)/(eta_yy-eta_xx)
    #Means
    bsmean_names = (f"E[{v}]" for v in analysed_columns)
    b2p_names = (f"{low_p}% ({v})" for v in analysed_columns)
    b97p_names = (f"{high_p}% ({v})" for v in analysed_columns)
    b50p_names = (f"50% ({v})" for v in analysed_columns)
    bs_var_names = (f"Var[{v}]" for v in analysed_columns)
    b2p_var_names = (f"{low_p}% (Var[{v}])" for v in analysed_columns)
    b97p_var_names = (f"{high_p}% (Var[{v}])" for v in analysed_columns)
    b50p_var_names = (f"50% (Var[{v}])" for v in analysed_columns)

    bs_mean_dist = np.mean(bootstrapped_data,axis=1)
    bs_means = np.mean(bs_mean_dist,axis=0)
    bs_var_dist = np.var(bootstrapped_data,axis=1)
    bs_vars = np.mean(bs_var_dist,axis=0)
    bs2p_means,bs_50p_means,bs97p_means = np.percentile(bs_mean_dist,[low_p,50,high_p],axis=0)
    bs2p_vars,bs_50p_vars,bs97p_vars = np.percentile(bs_var_dist,[low_p,50,high_p],axis=0)

    bsmeanres = zip(bsmean_names,bs_means)
    bsmean2pres = zip(b2p_names,bs2p_means)
    bsmean975pres = zip(b97p_names,bs97p_means)
    bsmean50pres = zip(b50p_names,bs_50p_means)
    bsvarres = zip(bs_var_names,bs_vars)
    bsvar2pres = zip(b2p_var_names,bs2p_vars)
    bsvar975pres = zip(b97p_var_names,bs97p_vars)
    bsvar50pres = zip(b50p_var_names,bs_50p_vars)

    #Calculate distributions for eta_xx,eta_yy,eta_xy, alongside avg+95CI
    meanx,meany,varx,vary,cxy = einsum_mean_cov_vars(bootstrapped_data[:,:,0],bootstrapped_data[:,:,1])
    eta_xx = varx/meanx**2
    eta_yy = vary/meany**2
    eta_xy = cxy/(meanx*meany)
    bs_exx_res = zip(["E[eta_xx]",f"{low_p}% (eta_xx)",f"50% (eta_xx)",f"{high_p}% (eta_xx)"],
                    [np.mean(eta_xx),*np.percentile(eta_xx,[low_p,50,high_p])])
    bs_eyy_res = zip(["E[eta_yy]",f"{low_p}% (eta_yy)",f"50% (eta_yy)",f"{high_p}% (eta_yy)"],
                    [np.mean(eta_yy),*np.percentile(eta_yy,[low_p,50,high_p])])
    bs_exy_res = zip(["E[eta_xy]",f"{low_p}% (eta_xy)",f"50% (eta_xy)",f"{high_p}% (eta_xy)"],
                    [np.mean(eta_xy),*np.percentile(eta_xy,[low_p,50,high_p])])

    #Use distribution from above to get following
    rhoref = lin_rho(sampleell,samplegg,avgfy,avgfx)
    cvref = lin_cvratio(sampleell,samplegg,avgfy,avgfx)
    #eta_xx/eta_yy (Cxy)
    CVxCVy = np.sqrt(eta_xx/eta_yy)
    bs_C_res = zip(["E[CVx/CVy]",f"{low_p}% (CVx/CVy)",f"50% (CVx/CVy)",f"{high_p}% (CVx/CVy)"],
                    [np.mean(CVxCVy),*np.percentile(CVxCVy,[low_p,50,high_p])])
    #eta_xx+eta_yy-2*eta_xy
    est_etabb = eta_xx+eta_yy-2*eta_xy
    bs_estetabb_res = zip(["E[eta_gg inf]",f"{low_p}% eta_gg inf",f"50% eta_gg inf",f"{high_p}% [eta_gg inf]"],
                    [np.mean(est_etabb),*np.percentile(est_etabb,[low_p,50,high_p])])
    #eta_xy/sqrt(eta_xx*eta_yy) (rhoxy_)
    rhoxy = eta_xy/np.sqrt(eta_xx*eta_yy)
    bs_rho_res = zip(["E[rhoxy]",f"{low_p}% (rhoxy)",f"50% (rhoxy)",f"{high_p}% (rhoxy)"],
                    [np.mean(rhoxy),*np.percentile(rhoxy,[low_p,50,high_p])])
    #rho/(CVx/CVy)
    fy_const_ratio = rhoxy/CVxCVy
    bs_fy_const_ratio = zip(["E[rho/CVratio]",f"{low_p}% (rho/CVratio)",f"50% (rho/CVratio)",f"{high_p}% (rho/CVratio)"],
                    [np.mean(fy_const_ratio),*np.percentile(fy_const_ratio,[low_p,50,high_p])])
    #rho/(CVy/CVx)
    fx_const_ratio = rhoxy*CVxCVy
    bs_fx_const_ratio = zip(["E[rho*CVratio]",f"{low_p}% (rho*CVratio)",f"50% (rho*CVratio)",f"{high_p}% (rho*CVratio)"],
                    [np.mean(fx_const_ratio),*np.percentile(fx_const_ratio,[low_p,50,high_p])])
    
    #deltarho
    delta_rho = rhoxy-rhoref
    bs_delta_rho_res = zip(["E[delta_rho]",f"{low_p}% (delta_rho)",f"50% (delta_rho)",f"{high_p}% (delta_rho)"],
                    [np.mean(delta_rho),*np.percentile(delta_rho,[low_p,50,high_p])])
    deltaCV = CVxCVy-cvref
    bs_deltaCV_res = zip(["E[deltaCV]",f"{low_p}% (deltaCV)",f"50% (deltaCV)",f"{high_p}% (deltaCV)"],
                    [np.mean(deltaCV),*np.percentile(deltaCV,[low_p,50,high_p])])
    #Inferred theta
    inftheta = (4/np.pi)*np.arctan2(eta_xy-eta_xx,eta_yy-eta_xy)
    bs_inftheta_res = zip(["E[theta_inf]",f"{low_p}% (theta_inf)",f"50% (theta_inf)",f"{high_p}% (theta_inf)"],
                    [np.mean(inftheta),*np.percentile(inftheta,[low_p,50,high_p])])
    #calcd theta
    calcdtheta = np.mean((4/np.pi)*np.arctan2(bootstrapped_data[:,:,-1],(1-bootstrapped_data[:,:,-2])),axis=1)
    bs_calcdtheta_res = zip(["E[theta_calc]",f"{low_p}% (theta_calc)",f"50% (theta_calc)",f"{high_p}% (theta_calc)"],
                    [np.mean(calcdtheta),*np.percentile(calcdtheta,[low_p,50,high_p])])

    
    
    analysed_data_dict.update(itertools.chain(bs_rho_res,bs_C_res,bsmeanres,bsmean2pres,bsmean975pres,bsmean50pres,
                                            bsvarres,bsvar2pres,bsvar975pres,bsvar50pres,
                                            bs_exx_res,bs_eyy_res,bs_exy_res,
                                            bs_estetabb_res,bs_fy_const_ratio,bs_fx_const_ratio,parammeanres,paramvarres,
                                            bs_delta_rho_res,bs_deltaCV_res,bs_inftheta_res,bs_calcdtheta_res))
    
    return analysed_data_dict

def pertexp_interpret_noparams(dataframe,
                           analysed_columns=['<x>','<y>',"<F_x>","<F_y>"],
                           low_p=2.5,high_p=97.5,nreps=3,nperts=9):
    """
    
    """
    analysed_data_dict = {}

    data_to_bs = dataframe[analysed_columns].to_numpy()
    #Generate bootstrap distribution of molecule and fb data values
    #bootstrapped_data = replicate_bootstrapping(data_to_bs,n_boots,n_reps=nreps,n_perts=nperts)
    #bootstrapped_data = bootstrap_gen(data_to_bs,n_boots)
    bootstrapped_data = full_bootstrap(data_to_bs)

    #Bootstrap to get means and 95%CI for the 
    #<x>,<y>,avgs of feedbacks,eta_xx/eta_yy,rho_xy,eta_xx,eta_yy,eta_xy,
    #eta_xx+eta_yy,(etaxy-etaxx)/(eta_yy-eta_xx)
    #Means
    bsmean_names = (f"E[{v}]" for v in analysed_columns)
    b2p_names = (f"{low_p}% ({v})" for v in analysed_columns)
    b97p_names = (f"{high_p}% ({v})" for v in analysed_columns)
    b50p_names = (f"50% ({v})" for v in analysed_columns)
    bs_var_names = (f"Var[{v}]" for v in analysed_columns)
    b2p_var_names = (f"{low_p}% (Var[{v}])" for v in analysed_columns)
    b97p_var_names = (f"{high_p}% (Var[{v}])" for v in analysed_columns)
    b50p_var_names = (f"50% (Var[{v}])" for v in analysed_columns)

    bs_mean_dist = np.mean(bootstrapped_data,axis=1)
    bs_means = np.mean(bs_mean_dist,axis=0)
    bs_var_dist = np.var(bootstrapped_data,axis=1)
    bs_vars = np.mean(bs_var_dist,axis=0)
    bs2p_means,bs_50p_means,bs97p_means = np.percentile(bs_mean_dist,[low_p,50,high_p],axis=0)
    bs2p_vars,bs_50p_vars,bs97p_vars = np.percentile(bs_var_dist,[low_p,50,high_p],axis=0)

    bsmeanres = zip(bsmean_names,bs_means)
    bsmean2pres = zip(b2p_names,bs2p_means)
    bsmean975pres = zip(b97p_names,bs97p_means)
    bsmean50pres = zip(b50p_names,bs_50p_means)
    bsvarres = zip(bs_var_names,bs_vars)
    bsvar2pres = zip(b2p_var_names,bs2p_vars)
    bsvar975pres = zip(b97p_var_names,bs97p_vars)
    bsvar50pres = zip(b50p_var_names,bs_50p_vars)

    #Calculate distributions for eta_xx,eta_yy,eta_xy, alongside avg+95CI
    meanx,meany,varx,vary,cxy = einsum_mean_cov_vars(bootstrapped_data[:,:,0],bootstrapped_data[:,:,1])
    eta_xx = varx/meanx**2
    eta_yy = vary/meany**2
    eta_xy = cxy/(meanx*meany)
    bs_exx_res = zip(["E[eta_xx]",f"{low_p}% (eta_xx)",f"50% (eta_xx)",f"{high_p}% (eta_xx)"],
                    [np.mean(eta_xx),*np.percentile(eta_xx,[low_p,50,high_p])])
    bs_eyy_res = zip(["E[eta_yy]",f"{low_p}% (eta_yy)",f"50% (eta_yy)",f"{high_p}% (eta_yy)"],
                    [np.mean(eta_yy),*np.percentile(eta_yy,[low_p,50,high_p])])
    bs_exy_res = zip(["E[eta_xy]",f"{low_p}% (eta_xy)",f"50% (eta_xy)",f"{high_p}% (eta_xy)"],
                    [np.mean(eta_xy),*np.percentile(eta_xy,[low_p,50,high_p])])

    #eta_xx/eta_yy (Cxy)
    CVxCVy = np.sqrt(eta_xx/eta_yy)
    bs_C_res = zip(["E[CVx/CVy]",f"{low_p}% (CVx/CVy)",f"50% (CVx/CVy)",f"{high_p}% (CVx/CVy)"],
                    [np.mean(CVxCVy),*np.percentile(CVxCVy,[low_p,50,high_p])])
    #eta_xx+eta_yy-2*eta_xy
    est_etabb = eta_xx+eta_yy-2*eta_xy
    bs_estetabb_res = zip(["E[eta_gg inf]",f"{low_p}% eta_gg inf",f"50% eta_gg inf",f"{high_p}% [eta_gg inf]"],
                    [np.mean(est_etabb),*np.percentile(est_etabb,[low_p,50,high_p])])
    #eta_xy/sqrt(eta_xx*eta_yy) (rhoxy_)
    rhoxy = eta_xy/np.sqrt(eta_xx*eta_yy)
    bs_rho_res = zip(["E[rhoxy]",f"{low_p}% (rhoxy)",f"50% (rhoxy)",f"{high_p}% (rhoxy)"],
                    [np.mean(rhoxy),*np.percentile(rhoxy,[low_p,50,high_p])])
    #rho/(CVx/CVy)
    fy_const_ratio = rhoxy/CVxCVy
    bs_fy_const_ratio = zip(["E[rho/CVratio]",f"{low_p}% (rho/CVratio)",f"50% (rho/CVratio)",f"{high_p}% (rho/CVratio)"],
                    [np.mean(fy_const_ratio),*np.percentile(fy_const_ratio,[low_p,50,high_p])])
    #rho/(CVy/CVx)
    fx_const_ratio = rhoxy*CVxCVy
    bs_fx_const_ratio = zip(["E[rho*CVratio]",f"{low_p}% (rho*CVratio)",f"50% (rho*CVratio)",f"{high_p}% (rho*CVratio)"],
                    [np.mean(fx_const_ratio),*np.percentile(fx_const_ratio,[low_p,50,high_p])])
    
    #Inferred theta
    inftheta = (4/np.pi)*np.arctan2(eta_xy-eta_xx,eta_yy-eta_xy)
    bs_inftheta_res = zip(["E[theta_inf]",f"{low_p}% (theta_inf)",f"50% (theta_inf)",f"{high_p}% (theta_inf)"],
                    [np.mean(inftheta),*np.percentile(inftheta,[low_p,50,high_p])])
    #calcd theta
    calcdtheta = np.mean((4/np.pi)*np.arctan2(bootstrapped_data[:,:,-1],(1-bootstrapped_data[:,:,-2])),axis=1)
    bs_calcdtheta_res = zip(["E[theta_calc]",f"{low_p}% (theta_calc)",f"50% (theta_calc)",f"{high_p}% (theta_calc)"],
                    [np.mean(calcdtheta),*np.percentile(calcdtheta,[low_p,50,high_p])])
    consistency = zip(['consistent fy count (stoch)','consistent fx count (stoch)'],[np.sum(np.sign(dataframe['<F_y>'])),np.sum(np.sign(1-dataframe['<F_x>']))])
    
    
    analysed_data_dict.update(itertools.chain(bs_rho_res,bs_C_res,bsmeanres,bsmean2pres,bsmean975pres,bsmean50pres,
                                            bsvarres,bsvar2pres,bsvar975pres,bsvar50pres,
                                            bs_exx_res,bs_eyy_res,bs_exy_res,
                                            bs_estetabb_res,bs_fy_const_ratio,bs_fx_const_ratio,bs_inftheta_res,bs_calcdtheta_res,consistency))
    
    return analysed_data_dict

class online_var:
    def __init__(self,array_size):
        self.n = 0
        self.mean = np.zeros(array_size)
        self.delta = np.zeros(array_size)
        self.delta2 = np.zeros(array_size)
        self.M2 = np.zeros(array_size)
        self.maximum_observed = np.full(array_size,np.nan,dtype=float)
        self.minimum_observed = np.full(array_size,np.nan,dtype=float)
    def update(self,x):
        self.n += 1
        np.subtract(x,self.mean,out=self.delta)
        self.mean += self.delta/self.n
        np.subtract(x,self.mean,out=self.delta2)
        self.M2 += np.multiply(self.delta,self.delta2)
        self.maximum_observed = np.fmax(self.maximum_observed,x)
        self.minimum_observed = np.fmin(self.minimum_observed,x)
    def var(self,ddof=0):
        return self.M2/(self.n-ddof)



def finitesample_interpret(dataframe,samplesize,replicates,rhoref,cvref,
                           analysed_columns=['<x>','<y>',"<F_x>","<F_y>"],
                           paramlist=['lambda','gamma'],
                           low_p=2.5,high_p=97.5):
    """
    Don't let my copy paste code fool you: this isn't bootstrapping a dataset but rather doing replicate draws from a larger dataset
    """
    analysed_data_dict = {}
    parammeans = dataframe[paramlist].mean()
    paramvars = dataframe[paramlist].var()
    parammeannames = (f"E[{v}]" for v in paramlist)
    paramvarnames = (f"Var[{v}]" for v in paramlist)
    parammeanres = zip(parammeannames,parammeans)
    paramvarres = zip(paramvarnames,paramvars)

    data_to_bs = dataframe[analysed_columns].to_numpy()
    #Generate bootstrap distribution of molecule and fb data values
    chosen_locs = np.concatenate([np.random.choice(len(data_to_bs), size=samplesize, replace=False) for rep in range(replicates)])
    bootstrapped_data = data_to_bs[chosen_locs].reshape(replicates, samplesize,len(analysed_columns))

    #bootstrapped_data = data_to_bs[np.random.choice(len(data_to_bs),size=(replicates,samplesize))]


    #Bootstrap to get means and 95%CI for the 
    #<x>,<y>,avgs of feedbacks,eta_xx/eta_yy,rho_xy,eta_xx,eta_yy,eta_xy,
    #eta_xx+eta_yy,(etaxy-etaxx)/(eta_yy-eta_xx)
    #Means
    bsmean_names = (f"E[{v}]" for v in analysed_columns)
    b2p_names = (f"{low_p}% ({v})" for v in analysed_columns)
    b97p_names = (f"{high_p}% ({v})" for v in analysed_columns)
    b50p_names = (f"50% ({v})" for v in analysed_columns)
    bs_mean_dist = np.mean(bootstrapped_data,axis=1)
    bs_means = np.mean(bs_mean_dist,axis=0)
    bs2p_means,bs_50p_means,bs97p_means = np.percentile(bs_mean_dist,[low_p,50,high_p],axis=0)
    bsmeanres = zip(bsmean_names,bs_means)
    bsmean2pres = zip(b2p_names,bs2p_means)
    bsmean975pres = zip(b97p_names,bs97p_means)
    bsmean50pres = zip(b50p_names,bs_50p_means)

    #Calculate distributions for eta_xx,eta_yy,eta_xy, alongside avg+95CI
    meanx,meany,varx,vary,cxy = einsum_mean_cov_vars(bootstrapped_data[:,:,0],bootstrapped_data[:,:,1])
    eta_xx = varx/meanx**2
    eta_yy = vary/meany**2
    eta_xy = cxy/(meanx*meany)
    bs_exx_res = zip(["E[eta_xx]",f"{low_p}% (eta_xx)",f"50% (eta_xx)",f"{high_p}% (eta_xx)"],
                    [np.mean(eta_xx),*np.percentile(eta_xx,[low_p,50,high_p])])
    bs_eyy_res = zip(["E[eta_yy]",f"{low_p}% (eta_yy)",f"50% (eta_yy)",f"{high_p}% (eta_yy)"],
                    [np.mean(eta_yy),*np.percentile(eta_yy,[low_p,50,high_p])])
    bs_exy_res = zip(["E[eta_xy]",f"{low_p}% (eta_xy)",f"50% (eta_xy)",f"{high_p}% (eta_xy)"],
                    [np.mean(eta_xy),*np.percentile(eta_xy,[low_p,50,high_p])])

    #Use distribution from above to get following
    
    #eta_xx/eta_yy (Cxy)
    CVxCVy = np.sqrt(eta_xx/eta_yy)
    bs_C_res = zip(["E[CVx/CVy]",f"{low_p}% (CVx/CVy)",f"50% (CVx/CVy)",f"{high_p}% (CVx/CVy)"],
                    [np.mean(CVxCVy),*np.percentile(CVxCVy,[low_p,50,high_p])])
    absdeltaCV = np.abs(CVxCVy-cvref)
    bs_abs_deltaCV_res = zip(["E[|deltaCV|]",f"{low_p}% (|deltaCV|)",f"50% (|deltaCV|)",f"{high_p}% (|deltaCV|)"],
                    [np.mean(absdeltaCV),*np.percentile(absdeltaCV,[low_p,50,high_p])])
    #eta_xx+eta_yy-2*eta_xy
    est_etabb = eta_xx+eta_yy-2*eta_xy
    bs_estetabb_res = zip(["E[eta_gg inf]",f"{low_p}% eta_gg inf",f"50% eta_gg inf",f"{high_p}% [eta_gg inf]"],
                    [np.mean(est_etabb),*np.percentile(est_etabb,[low_p,50,high_p])])
    #eta_xy/sqrt(eta_xx*eta_yy) (rhoxy_)
    rhoxy = eta_xy/np.sqrt(eta_xx*eta_yy)
    bs_rho_res = zip(["E[rhoxy]",f"{low_p}% (rhoxy)",f"50% (rhoxy)",f"{high_p}% (rhoxy)"],
                    [np.mean(rhoxy),*np.percentile(rhoxy,[low_p,50,high_p])])
    absdelta_rho = np.abs(rhoxy-rhoref)
    bs_abs_delta_rho_res = zip(["E[|deltarho|]",f"{low_p}% (|deltarho|)",f"50% (|deltarho|)",f"{high_p}% (delta_rho)"],
                    [np.mean(absdelta_rho),*np.percentile(absdelta_rho,[low_p,50,high_p])])
    
    delta_rho = rhoxy-rhoref
    bs_delta_rho_res = zip(["E[delta_rho]",f"{low_p}% (delta_rho)",f"50% (delta_rho)",f"{high_p}% (delta_rho)"],
                    [np.mean(delta_rho),*np.percentile(delta_rho,[low_p,50,high_p])])
    deltaCV = CVxCVy-cvref
    bs_deltaCV_res = zip(["E[deltaCV]",f"{low_p}% (deltaCV)",f"50% (deltaCV)",f"{high_p}% (deltaCV)"],
                    [np.mean(deltaCV),*np.percentile(deltaCV,[low_p,50,high_p])])

    distance_to_ref = np.sqrt((deltaCV)**2+(delta_rho)**2)
    bs_distance_to_ref_res = zip(["E[distance to ref]",f"{low_p}% (distance to ref)",f"50% (distance to ref)",f"{high_p}% (distance to ref)"],
                    [np.mean(distance_to_ref),*np.percentile(distance_to_ref,[low_p,50,high_p])])
    #correct Fy sign fraction
    fy_inf_dist = eta_xy-eta_xx
    fy_low_inf,fy_high_inf = np.percentile(fy_inf_dist,[low_p,high_p])
    fy_low_sign,fy_high_sign = np.sign(fy_low_inf),np.sign(fy_high_inf)
    fy_correct_hits = ((np.sign(bs_means[-1])==0)&(np.sign(fy_low_sign)*np.sign(fy_high_sign)==-1)) | (np.sign(bs_means[-1])==np.sign(fy_inf_dist))
    fy_fraction_correct_res = zip(["Fy fraction correct"],
                    [np.mean(fy_correct_hits)])


    analysed_data_dict.update(itertools.chain(bs_rho_res,bs_C_res,bsmeanres,bsmean2pres,bsmean975pres,bsmean50pres,
                                            bs_exx_res,bs_eyy_res,bs_exy_res,
                                            bs_estetabb_res,parammeanres,paramvarres,
                                            bs_deltaCV_res,bs_delta_rho_res,
                                            bs_abs_deltaCV_res,bs_abs_delta_rho_res,
                                            bs_distance_to_ref_res,fy_fraction_correct_res))
    return analysed_data_dict

def finitesample_bootstrap(dataframe,samplesize,n_boots,rhoref,cvref,
                           analysed_columns=['<x>','<y>',"<F_x>","<F_y>"],
                           paramlist=['lambda','gamma'],
                           low_p=2.5,high_p=97.5):
    """
    Don't let my copy paste code fool you: this isn't bootstrapping a dataset but rather doing replicate draws from a larger dataset,
    then bootstrapping them
    """
    analysed_data_dict = {}
    parammeans = dataframe[paramlist].mean()
    paramvars = dataframe[paramlist].var()
    parammeannames = (f"E[{v}]" for v in paramlist)
    paramvarnames = (f"Var[{v}]" for v in paramlist)
    parammeanres = zip(parammeannames,parammeans)
    paramvarres = zip(paramvarnames,paramvars)

    data_to_bs = dataframe[analysed_columns].to_numpy()
    #Choose a finite sample
    data_to_bs = data_to_bs[np.random.choice(len(data_to_bs), size=samplesize, replace=False)]
    #bootstrap that finite sample
    bootstrapped_data = bootstrap_gen(data_to_bs,n_boots)

    #bootstrapped_data = data_to_bs[np.random.choice(len(data_to_bs),size=(replicates,samplesize))]


    #Bootstrap to get means and 95%CI for the 
    #<x>,<y>,avgs of feedbacks,eta_xx/eta_yy,rho_xy,eta_xx,eta_yy,eta_xy,
    #eta_xx+eta_yy,(etaxy-etaxx)/(eta_yy-eta_xx)
    #Means
    bsmean_names = (f"E[{v}]" for v in analysed_columns)
    b2p_names = (f"{low_p}% ({v})" for v in analysed_columns)
    b97p_names = (f"{high_p}% ({v})" for v in analysed_columns)
    b50p_names = (f"50% ({v})" for v in analysed_columns)
    bs_mean_dist = np.mean(bootstrapped_data,axis=1)
    bs_means = np.mean(bs_mean_dist,axis=0)
    bs2p_means,bs_50p_means,bs97p_means = np.percentile(bs_mean_dist,[low_p,50,high_p],axis=0)
    bsmeanres = zip(bsmean_names,bs_means)
    bsmean2pres = zip(b2p_names,bs2p_means)
    bsmean975pres = zip(b97p_names,bs97p_means)
    bsmean50pres = zip(b50p_names,bs_50p_means)

    #Calculate distributions for eta_xx,eta_yy,eta_xy, alongside avg+95CI
    meanx,meany,varx,vary,cxy = einsum_mean_cov_vars(bootstrapped_data[:,:,0],bootstrapped_data[:,:,1])
    eta_xx = varx/meanx**2
    eta_yy = vary/meany**2
    eta_xy = cxy/(meanx*meany)
    bs_exx_res = zip(["E[eta_xx]",f"{low_p}% (eta_xx)",f"50% (eta_xx)",f"{high_p}% (eta_xx)"],
                    [np.mean(eta_xx),*np.percentile(eta_xx,[low_p,50,high_p])])
    bs_eyy_res = zip(["E[eta_yy]",f"{low_p}% (eta_yy)",f"50% (eta_yy)",f"{high_p}% (eta_yy)"],
                    [np.mean(eta_yy),*np.percentile(eta_yy,[low_p,50,high_p])])
    bs_exy_res = zip(["E[eta_xy]",f"{low_p}% (eta_xy)",f"50% (eta_xy)",f"{high_p}% (eta_xy)"],
                    [np.mean(eta_xy),*np.percentile(eta_xy,[low_p,50,high_p])])
    #Use distribution from above to get following
    
    #eta_xx/eta_yy (Cxy)
    CVxCVy = np.sqrt(eta_xx/eta_yy)
    bs_C_res = zip(["E[CVx/CVy]",f"{low_p}% (CVx/CVy)",f"50% (CVx/CVy)",f"{high_p}% (CVx/CVy)"],
                    [np.mean(CVxCVy),*np.percentile(CVxCVy,[low_p,50,high_p])])
    absdeltaCV = np.abs(CVxCVy-cvref)
    bs_abs_deltaCV_res = zip(["E[|deltaCV|]",f"{low_p}% (|deltaCV|)",f"50% (|deltaCV|)",f"{high_p}% (|deltaCV|)"],
                    [np.mean(absdeltaCV),*np.percentile(absdeltaCV,[low_p,50,high_p])])
    #eta_xx+eta_yy-2*eta_xy
    est_etabb = eta_xx+eta_yy-2*eta_xy
    bs_estetabb_res = zip(["E[eta_gg inf]",f"{low_p}% eta_gg inf",f"50% eta_gg inf",f"{high_p}% [eta_gg inf]"],
                    [np.mean(est_etabb),*np.percentile(est_etabb,[low_p,50,high_p])])
    #eta_xy/sqrt(eta_xx*eta_yy) (rhoxy_)
    rhoxy = eta_xy/np.sqrt(eta_xx*eta_yy)
    bs_rho_res = zip(["E[rhoxy]",f"{low_p}% (rhoxy)",f"50% (rhoxy)",f"{high_p}% (rhoxy)"],
                    [np.mean(rhoxy),*np.percentile(rhoxy,[low_p,50,high_p])])
    absdelta_rho = np.abs(rhoxy-rhoref)
    bs_abs_delta_rho_res = zip(["E[|deltarho|]",f"{low_p}% (|deltarho|)",f"50% (|deltarho|)",f"{high_p}% (delta_rho)"],
                    [np.mean(absdelta_rho),*np.percentile(absdelta_rho,[low_p,50,high_p])])
    
    delta_rho = rhoxy-rhoref
    bs_delta_rho_res = zip(["E[delta_rho]",f"{low_p}% (delta_rho)",f"50% (delta_rho)",f"{high_p}% (delta_rho)"],
                    [np.mean(delta_rho),*np.percentile(delta_rho,[low_p,50,high_p])])
    deltaCV = CVxCVy-cvref
    bs_deltaCV_res = zip(["E[deltaCV]",f"{low_p}% (deltaCV)",f"50% (deltaCV)",f"{high_p}% (deltaCV)"],
                    [np.mean(deltaCV),*np.percentile(deltaCV,[low_p,50,high_p])])

    distance_to_ref = np.sqrt((deltaCV)**2+(delta_rho)**2)
    bs_distance_to_ref_res = zip(["E[distance to ref]",f"{low_p}% (distance to ref)",f"50% (distance to ref)",f"{high_p}% (distance to ref)"],
                    [np.mean(distance_to_ref),*np.percentile(distance_to_ref,[low_p,50,high_p])])
    #correct Fy sign fraction
    fy_inf_dist = eta_xy-eta_xx
    fy_low_inf,fy_high_inf = np.percentile(fy_inf_dist,[low_p,high_p])
    fy_low_sign,fy_high_sign = np.sign(fy_low_inf),np.sign(fy_high_inf)
    fy_correct_hits = ((np.sign(bs_means[-1])==0)&(np.sign(fy_low_sign)*np.sign(fy_high_sign)==-1)) | (np.sign(bs_means[-1])==np.sign(fy_inf_dist))
    fy_fraction_correct_res = zip(["Fy fraction correct"],
                    [np.mean(fy_correct_hits)])

    analysed_data_dict.update(itertools.chain(bs_rho_res,bs_C_res,bsmeanres,bsmean2pres,bsmean975pres,bsmean50pres,
                                            bs_exx_res,bs_eyy_res,bs_exy_res,
                                            bs_estetabb_res,parammeanres,paramvarres,
                                            bs_deltaCV_res,bs_delta_rho_res,
                                            bs_abs_deltaCV_res,bs_abs_delta_rho_res,
                                            bs_distance_to_ref_res,fy_fraction_correct_res))

    return analysed_data_dict

def rho_bars(bsdata):
    return [bsdata['E[rhoxy]']-bsdata['2.5% (rhoxy)'],bsdata['97.5% (rhoxy)']-bsdata['E[rhoxy]']]
def CV_bars(bsdata):
    return [bsdata['E[CVx/CVy]']-bsdata['2.5% (CVx/CVy)'],bsdata['97.5% (CVx/CVy)']-bsdata['E[CVx/CVy]']]
def categorize_scaled_metric(metric):
    return np.where(metric.between(0,1,inclusive='neither'),'A',
                    np.where(metric.between(-2,0,inclusive='neither'),'B',
                             np.where(metric.between(-3,-2,inclusive='neither'),'C',
                                      np.where(metric == 0,'D',
                                               np.where(metric == -2,'E','uncategorized')))))