# -*- coding: utf-8 -*-
"""
Module of functions for iterative meSMLM VTI computation

imgparams() reports values for parameters related to the imaging system
estimands() reports values for to-be-estimated parameters
VTIparams() reports values for parameters needed to evaluate the VTI
intensity(x, phi) returns the illumination intensity of a sinusoidal pattern for a given coordinate and phase shift
MC_sample(mean, cov) returns sampled parameter values of the simple MC-sampling distibution (here, a Gaussian)
FIM(mod) returns the Fisher information matrix
VTI_prior(mean, cov, samples) returns the Bayesian prior information matrix, assuming a Gaussian prior
VTI_data(mod, samples) returns the Bayesian data information matrix
VTI_eval(mean, cov, mod) returns a simple Monte Carlo estimate of the Van Trees inequality, given prior information mean and cov and imaging configuration contained in mod
VTI_theoretical(iteration, mod, theta_x, theta_prev=None, sigma_prev=None) returns the theoretical (analytical approximation) Van Trees inequality after each iteration, given an imaging configuration contained in mod
MAP_eval(mod_per_spot) returns an array of maximum a posteriori estimates under given imaging conditions, as well as the Cramér-Rao lower bound and the Fisher information matrix
"""

import numpy as np
from photonpy.cpp.context import Context
from photonpy.cpp.simflux import SIMFLUX

# =============================================================================
# 
#                                   imgparams
# 
# =============================================================================

def imgparams():
    '''
    imgparams() reports values for parameters related to the imaging system
    
    Output:
        roisize: Amount of pixels in 1 direction (pixels)
        m: Modulation contrast of the illumination patterns
        kx: Spatial pattern x-frequency (rad/pixels)
        ky: Spatial pattern y-frequency (rad/pixels)
        sigmap: Gaussian PSF standard deviation (pixels)
        
        hidden: Vector of parameters that are not explicitly used, respectively:
            delta_x: Pixel size of square pixels (m)
            p: Illumination pattern pitch (m)
            omega: Spatial pattern frequency (rad/m)
            L: Wavelength of illumination light (m)
            NA: Numerical aperture
            sigma: Gaussian PSF standard deviation (m)
    '''
    
    #Camera parameters
    roisize = 11; #Amount of pixels in 1 direction
    delta_x = 65*10**-9; #Pixel size of square pixels
    
    #Illumination pattern parameters
    m = 0.95; #Modulation contrast of the illumination patterns
    p = 243.75*10**-9; #Illumination pattern pitch
    omega = 2*np.pi/p; #Spatial pattern frequency
    kx = omega * delta_x; #Spatial pattern x-frequency
    ky = omega * delta_x; #Spatial pattern y-frequency

    #Other
    L = 680*10**(-9); #Wavelength of emission light
    NA = 1.41; #Numerical aperture
    sigma = L/(4*NA); #Gaussian PSF standard deviation
    sigmap=sigma/delta_x; #Gaussian PSF standard deviation
    
    hidden = [delta_x,p,omega,L,NA,sigma] #Vector of parameters that are not explicitly used
    
    return roisize,m,kx,ky,sigmap,hidden

# =============================================================================
# 
#                                   estimands
# 
# =============================================================================

def estimands():
    '''
    estimands() reports values for to-be-estimated parameters
    
    Output:
        theta: vector of estimands, respectively:
            theta_x: average emitter x-position relative to first pixel, before subpixel randomization (pixels)
            theta_y: average emitter y-position relative to first pixel, before subpixel randomization (pixels)
            theta_I: expected signal photon intensity over all iterations under maximum illumination (photons)
            theta_b: expected background photon count per pixel, assumed to be uniform over the region of interest (photons/pixel)
    '''

    roisize,_,_,_,_,_ = imgparams();
    
    theta_x = roisize/2; #Emitter x-position relative to first pixel
    theta_y = roisize/2; #Emitter y-position relative to first pixel
    theta_I = 2000; #Expected signal photon intensity over all iterations under maximum illumination
    theta_b = 8; #Expected background photon count per pixel, assumed to be uniform over the region of interest
    
    theta=np.array([[theta_x, theta_y, theta_I, theta_b]]); #Vector of estimands
    
    return theta

# =============================================================================
# 
#                                   VTIparams
# 
# =============================================================================

def VTIparams():
    '''
    VTIparams() reports values for parameters needed to evaluate the VTI
    
    Output:
        MC_samples: Amount of samples used for simple Monte Carlo estimation of the VTI
        itermax: Amount of simulated VTI iterations
        alpha: Aggressiveness parameter, that is the amount of standard deviations between the emitter estimate and a pattern minimum
    '''
    
    MC_samples = 50000; #Amount of samples used for simple Monte Carlo estimation
    itermax = 3; #Amount of imeSMLM iterations
    alpha = 3; #Aggressiveness parameter, that is the amount of standard deviations between the emitter estimate and a pattern minimum
    
    return MC_samples,itermax,alpha


# =============================================================================
# 
#                                   intensity
# 
# =============================================================================

def intensity(x, phi):
    '''
    intensity(x, phi) returns the illumination intensity of a sinusoidal pattern for a given coordinate and phase shift
    
    Input:
        x:      Relative spatial coordinate in the direction of the sinusoid, in pixels
        phi:    Phase shift of the sinusoidal pattern with respect to the first pixel, in rad
    
    Output:
        intensity:  Illumination pattern intensity between 0 and 1, dimensionless
    '''
    
    _,m,kx,_,_,other = imgparams()
    intensity = 0.5*(1+m*np.sin(kx*x-phi))
    
    return intensity

# =============================================================================
# 
#                                   MC_sample
# 
# =============================================================================

def MC_sample(mean, cov, bgcompensation=True):
    '''
    MC_sample(mean, cov) returns sampled parameter values of the simple MC-sampling distibution (here, a Gaussian)
    
    Input:
        mean:   numpy 1D-array (1 x size) of true or estimated estimand values
        cov:    numpy 2D-array (size x size) of covariance matrix of prior distribution
    
    Optional input:    
        bgcompensation: Boolean, set to True to force that background coefficient is non-negative
    
    Output:
        sample:  numpy array (MC_samples x size) of sampled parameter values
    '''
    
    MC_samples,_,_ = VTIparams();
    size = np.size(mean)
    transformationmatrix = np.linalg.cholesky(cov); #Cholesky decomposition cov = transformationmatrix @ transformationmatrix.T
    samples = mean + (transformationmatrix@np.random.randn(size,MC_samples)).T;
    
    if bgcompensation==True:
        indices = samples[:,3]<0
        samples[indices,3] = 0;
    
    return samples

# =============================================================================
# 
#                                    FIM
# 
# =============================================================================

def FIM(mod):
    '''
    FIM(mod) returns the Fisher information matrix
    
    Input:
        mod: numpy array of imaging configuration, given as ...
            ([spatial pattern x-frequency (px), spatial pattern y-frequency (px), spatial pattern z-frequency (px), ...
            modulation contrast, pattern phase, relative intensity], [...], etc. for each pattern)
        
    Output:
        fim: Fisher information matrix
    '''    
    
    roisize,_,_,_,sigma,_ = imgparams();
    theta = estimands();
    
    with Context() as ctx:
        s = SIMFLUX(ctx)
        sf_psf = s.CreateEstimator_Gauss2D(sigma, len(mod), roisize, len(mod), True) #Simflux PSF object
        fim = sf_psf.FisherMatrix(theta,constants=mod)
    
    return fim  
  
# =============================================================================
# 
#                                   VTI_prior
# 
# =============================================================================
    
def VTI_prior(mean, cov, samples):
    '''
    VTI_prior(mean, cov, samples) returns the Bayesian prior information matrix, assuming a Gaussian prior
    
    Input:
        mean: numpy 1D-array (1 x size) of true or estimated estimand values
        cov: numpy 2D-array (size x size) of covariance matrix of prior distribution
        samples: numpy array (MC_samples x size) of sampled parameter values
        
    Output:
        JP: Bayesian prior information matrix
    '''
    MC_samples,_,_ = VTIparams();
    invcov = np.linalg.inv(cov);
    
    diff = samples-mean;
    FPI = (invcov@diff.T) @ ((invcov@diff.T).T); #Fisher information contained in prior distribution
    JP = FPI/MC_samples
    
    return JP

# =============================================================================
# 
#                                   VTI_data
# 
# =============================================================================

def VTI_data(mod, samples):
    '''
    VTI_data(mod, samples) returns the Bayesian data information matrix
    
    Input:
        mod: numpy array of imaging configuration, given as ...
            ([spatial pattern x-frequency (px), spatial pattern y-frequency (px), spatial pattern z-frequency (px), ...
            modulation contrast, pattern phase, relative intensity], [...], etc. for each pattern)
        samples: numpy array (MC_samples x size) of sampled parameter values
        
    Output:
        JD: Bayesian data information matrix
    '''    
    indices = samples[:,3] < 0
    if max(indices)==True:
        print('Warning: negative background coefficient found in samples. Found JD might be inaccurate. Consider resampling with bgcompensation=True')
    
    roisize,_,_,_,sigma,_ = imgparams();
    MC_samples,_,_ = VTIparams();
    
    mod_per_spot = np.repeat([mod],MC_samples,0)
    
    with Context() as ctx:
        s = SIMFLUX(ctx)
        sf_psf = s.CreateEstimator_Gauss2D(sigma, len(mod), roisize, len(mod), True) #Simflux PSF object
        I = sf_psf.FisherMatrix(samples,constants=mod_per_spot)
    
    JD = np.mean(I,axis=0)
    return JD

# =============================================================================
# 
#                                   VTI_eval
# 
# =============================================================================

def VTI_eval(mean, cov, mod, bgcompensation=True):
    '''
    VTI_eval(mean, cov, mod) returns a simple Monte Carlo estimate of the Van Trees inequality, 
    given prior information mean and cov and imaging configuration contained in mod
    
    Input:
        mean: numpy array of true or estimated estimand values
        cov: numpy 2D-array (size x size) of covariance matrix of prior distribution
        mod: numpy array of imaging configuration, given as ...
            ([spatial pattern x-frequency (px), spatial pattern y-frequency (px), spatial pattern z-frequency (px), ...
            modulation contrast, pattern phase, relative intensity], [...], etc. for each pattern)
        
    Optional input:
        bgcompensation: Boolean, set to True to force that background coefficient for sampled estimands is non-negative
            
    Output:
        VTI: mean-squared error matrix as given by the VTI
    '''
    
    samples = MC_sample(mean,cov,bgcompensation);
    
    JP = VTI_prior(mean,cov,samples);
    JD = VTI_data(mod,samples);
        
    VTI = JD + JP;

    return VTI

# =============================================================================
# 
#                                   VTI_theoretical
# 
# =============================================================================

def VTI_theoretical(iteration, mod, theta_prev=None, sigma_prev=None):
    '''
    VTI_theoretical(iteration, mod, theta_x, theta_prev=None, sigma_prev=None) 
    returns the theoretical (analytical approximation) Van Trees inequality after each iteration, 
    given an imaging configuration contained in mod
    
    Assumptions:
        - 1D localization
        - Perfect modulation (m = 1)
        - No background (theta_b = 0)
        - No pixelation
        
    Note:
        This function assumes that mod originates from 2D-localization, using 4 - 4 - 4 - ... - 4 patterns per iteration.
        It therefore disregards patterns corresponding to the y-direction.
    
    Input:
        iteration: iteration number
        mod: numpy array of imaging configuration, given as ...
            ([spatial pattern x-frequency (px), spatial pattern y-frequency (px), spatial pattern z-frequency (px), ...
            modulation contrast, pattern phase, relative intensity], [...], etc. for each pattern)
        theta_prev: if iteration > 0, theta_prev is the position estimate from the previous iteration (px)
        sigma_prev: if iteration > 0, sigma_prev is the VTI/CRLB from the previous iteration (px**2)
            
    Output:
        VTI: mean-squared error matrix as given by the VTI
    '''
    #Load image configuration parameters
    _,_,kx,_,sigmap,_=imgparams();
    ck = mod[0,5]
    phixp = mod[0,4]
    phixm = mod[1,4]
    theta = estimands();
    thetax = theta[0,0]
    thetaI = theta[0,2]
    
    if iteration == 0:
        #Compute Fisher information
        VTI = thetaI*ck*kx**2*(1- ( np.cos(kx*thetax-phixp) + np.cos(kx*thetax-phixm) ) ) + thetaI*ck/sigmap**2*(1+( np.cos(kx*thetax-phixp) + np.cos(kx*thetax-phixm) ))
    
    elif iteration >= 1:
        #Prior information JP
        JP = 1/(sigma_prev**2);
        
        #Initialize and compute JD
        JD=0;
        
        for config in range(len(mod)):
            if config < len(mod)/2: #Only evaluate x-patterns
                JD += theta[0,2]*2*mod[config,-1]*(kx)**2*(1-np.cos(kx*theta_prev-mod[config,-2])*np.exp(-0.5*kx**2*sigma_prev**2)) + theta[0,2]*2*mod[config,-1]/(sigmap**2)*(1+np.cos(kx*theta_prev-mod[config,-2])*np.exp(-0.5*kx**2*sigma_prev**2))
        
        #Bayesian information matrix
        VTI = JP + JD;
   
    return VTI

# =============================================================================
# 
#                                   MAP_eval
# 
# =============================================================================

def MAP_eval(mod_per_spot, theta_per_spot, previousData=np.zeros(1), rejectUnconverged=False,limitsUnconverged=[[0,2],[0,2],[1/3,3],[0,3]], maxcountUnconverged = 10000):
    '''
    MAP_eval(mod_per_spot) returns an array of maximum a posteriori estimates (with uniform initial prior) under given imaging conditions, ...
        as well as the Cramér-Rao lower bound and the Fisher information matrix
    
    Input:
        mod_per_spot: numpy 3D-array (MAP repetitions x amount of pattern configurations x pattern configuration parameters) containing imaging conditions
    
    Optional input:
        previousData: numpy 3D-array, containing samples from earlier iterations
        rejectUnconverged: Boolean, set to True to resample and re-evaluate unconverged MAP estimates
        limitsUnconverged: list of multipliers. If MAP lies outside of [min,max]*theta, it is considered unconverged and rejected.
        maxcountUnconverged: maximum amount of resamples and re-evaluations of MAP estimate, safeguard in case limitsUnconverged is too strict.
    
    Output:
        estim:  numpy 2D-array (MAP repetitions x amount of parameters) of MAP estimates
        crlb:   numpy 2D-array (MAP repetitions x amount of parameters) of CRLB values
        fi:     numpy 3D-array (MAP repetitions x amount of parameters x amount of parameters) of Fisher information matrices for each MAP repetition
    '''
    #Load parameters
    roisize,_,_,_,sigmap,_ = imgparams();
     
    #When enabled, checks if any MAP estimates did not converge and if so, resamples data and then recomputes them  
    if rejectUnconverged==True:
        
        #Allocation
        smp = np.zeros([len(mod_per_spot),len(mod_per_spot[0,:,:]), roisize, roisize])
        estim = np.zeros([len(mod_per_spot), len(theta_per_spot[0,:]) ])
        crlb = np.zeros([len(mod_per_spot), len(theta_per_spot[0,:]) ])
        fi = np.zeros([len(mod_per_spot), len(theta_per_spot[0,:]), len(theta_per_spot[0,:])])
        unconverged = np.array([True for i in range(len(mod_per_spot))])
        
        #Counter if limits are too strict
        counter = 0
        
        while max(unconverged)==True and counter <= maxcountUnconverged:
            counter += 1;
            
            #Generate PSF, data and compute MAP and CRLB
            with Context() as ctx:
                #Create photonpy SIMFLUX PSF object
                s = SIMFLUX(ctx)
                sf_psf = s.CreateEstimator_Gauss2D(sigmap, len(mod_per_spot[0,:,:]), roisize, len(mod_per_spot[0,:,:]), True)
                sf_psf.lmparams.iterations = 500
                sf_psf.lmparams.factor = 2
                
                #Generate Poisson data from PSF, then estimate using MAP
                if previousData.any() == False:
                    smp[unconverged,:,:,:] = sf_psf.GenerateSample(theta_per_spot[unconverged,:], constants=mod_per_spot[unconverged,:,:]) #Generate poisson data from simflux PSF
                else:
                    if counter == 1:
                        smp[:,0:len(previousData[0]),:,:] = previousData
                    sf_psf_1it = s.CreateEstimator_Gauss2D(sigmap, len(mod_per_spot[0,len(previousData[0])::,:]), roisize, len(mod_per_spot[0,len(previousData[0])::,:]), True)
                    smp[unconverged,len(previousData[0])::,:,:] = sf_psf_1it.GenerateSample(theta_per_spot[unconverged,:], constants=mod_per_spot[unconverged,len(previousData[0])::,:]) #Generate poisson data from simflux PSF
                
                estim[unconverged,:],_,_ = sf_psf.Estimate(smp[unconverged,:,:,:],constants=mod_per_spot[unconverged,:,:]) #estim: MAP for theta

                #Compute CRLB and FIM
                crlb[unconverged,:] = sf_psf.CRLB(estim[unconverged,:],constants=mod_per_spot[unconverged,:,:]) #SIMFLUX CRLB: (amount of emitters, amount of parameters)
                fi[unconverged,:,:] = sf_psf.FisherMatrix(estim[unconverged,:],constants=mod_per_spot[unconverged,:,:])
        
            #Check if any MAP estimate did not converge and find indices
            unconvergedxmin = estim[:,0] <= theta_per_spot[:,0] - roisize//3;  
            unconvergedxmax = estim[:,0] >= theta_per_spot[:,0] + roisize//3; 
            
            unconvergedymin = estim[:,1] <= theta_per_spot[:,1] - roisize//3;
            unconvergedymax = estim[:,1] >= theta_per_spot[:,1] + roisize//3;
            
            unconvergedImin = estim[:,2] < limitsUnconverged[2][0]*theta_per_spot[:,2] 
            unconvergedImax = estim[:,2] > limitsUnconverged[2][1]*theta_per_spot[:,2] 
            
            unconvergedbmin = estim[:,3] < limitsUnconverged[3][0]*theta_per_spot[:,3] 
            unconvergedbmax = estim[:,3] > limitsUnconverged[3][1]*theta_per_spot[:,3]
            
            unconverged = unconvergedxmin | unconvergedxmax | unconvergedymin | unconvergedymax | unconvergedImin | unconvergedImax | unconvergedbmin | unconvergedbmax
            
            if counter==maxcountUnconverged and max(unconverged)==True:
                print('Maximum amount of MAP re-evaluations reached, but MAP did not converge.')
        
    #Generate PSF, data and compute MAP and CRLB
    else:
        with Context() as ctx:
            #Create photonpy SIMFLUX PSF object
            s = SIMFLUX(ctx)
            sf_psf = s.CreateEstimator_Gauss2D(sigmap, len(mod_per_spot[0,:,:]), roisize, len(mod_per_spot[0,:,:]), True)
            sf_psf.lmparams.iterations = 5000
            
            #Generate Poisson data from PSF, then estimate using MAP
            smp = sf_psf.GenerateSample(theta_per_spot, constants=mod_per_spot) #Generate poisson data from simflux PSF
            estim,_,_ = sf_psf.Estimate(smp,constants=mod_per_spot) #estim: MAP for theta
            
            #Compute CRLB and FIM
            crlb = sf_psf.CRLB(estim,constants=mod_per_spot) #SIMFLUX CRLB: (amount of emitters, amount of parameters)
            fi = sf_psf.FisherMatrix(estim,constants=mod_per_spot)
    
    return estim,crlb,fi,smp