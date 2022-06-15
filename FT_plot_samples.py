# -*- coding: utf-8 -*-
"""
Reading data, processing and plotting of Figures
    3a, 3b, 3c
"""
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.optimize import least_squares
import pandas as pd
import VTI_helper as vti
import matplotlib as mpl

new_rc_params = {
"font.size": 12}
mpl.rcParams.update(new_rc_params)

def readdata(setnumber, subsets):
    '''
    Reads VTI and MAP data from ./SimData/
    
    Input:
        setnumber: number of dataset as given in Datasets.xlsx
        subsets: amount of subsets used for RMSE computation
        
    Output:
        Jinv: Inverted Bayesian information matrix
        thetaMAP: stack of MAP estimates for dataset
        smp: regions of interest
    '''
    loadpath = './SimData/'

    datasets_pd = pd.read_excel('Datasets.xlsx')
    datasets = np.array(datasets_pd)
    
    roisize,m,_,_,_,other = vti.imgparams();
    
    # Excel numbering starts at 1
    setnumber -= 1
    
    itermax = int(datasets[setnumber,1]);
    alpha = int(datasets[setnumber,2]);
    thetaI = datasets[setnumber,3];
    m = datasets[setnumber,4];
    thetab = datasets[setnumber,5];
    
    saveJ = 'FT-J-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m))+ '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    savethetaMAP = 'FT-thetaMAP-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m))+ '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    savesmp ='FT-smp-iter-' + str(itermax) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab)) +'.npy'
    savetheta = 'FT-theta-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m))+ '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    
    J = np.load(loadpath+saveJ)
    Jinv = np.linalg.inv(J)
    thetaMAP = np.load(loadpath+savethetaMAP)
    smp = np.load(loadpath + savesmp)
    theta_per_spot = np.load(loadpath+savetheta)
    noise = theta_per_spot - np.array([5.5, 5.5, thetaI, thetab])
    
    for i in range(len(thetaMAP[0])):
        thetaMAP[:,i,:] -= noise    
    
    return Jinv, thetaMAP, smp

def model(x, alpha):
    '''
    Model used for fitting a Gaussian on a histogram
    
    Input:
        x: array of fitting parameters (i.e. height, standard deviation)
        alpha: coordinate to describe the center of a bin of a histogram
        
    Output:
        gaussian: value of the Gaussian at coordinate alpha
    '''
    gaussian = x[0] * np.exp(-alpha**2 / (2 * x[1]**2))
    return gaussian

def error(x, alpha, h): 
    '''
    Difference between the simulated data and the Gaussian function.
    That is, the difference between the center of each bin of the histogram and 
    the corresponding coordinate of the Gaussian density function.
    
    Input:
        x: array of fitting parameters (i.e. height, standard deviation)
        alpha: coordinate to describe the center of a bin of a histogram
        h: height of the histogram at coordinate alpha
        
    Output:
        diffrence: difference between simulated data and the Gaussian function
    '''
    difference = h - model(x, alpha)
    return difference

def jac(x, alpha, h):
    '''
    Jacobian, i.e. the partial derivatives of the difference between the simulated data 
    and the Gaussian function, with respect to the parameters contained in x.
    
    Input:
        x: array of fitting parameters (i.e. height, standard deviation)
        alpha: coordinate to describe the center of a bin of a histogram
        h: height of the histogram at coordinate alpha
        
    Output:
        J: Jacobian
    '''
    J = np.empty((alpha.size, x.size))
    J[:, 0] = -np.exp(-alpha**2 / (2 * x[1]**2))
    J[:, 1] = -x[0] * np.exp(-alpha**2 / (2 * x[1]**2)) * (alpha**2 /(x[1]**3))
    return J

def roundup(x,ndigits):
    '''
    Rounds up the input x up to ndigits digits.
    
    Input:
        x: value to be rounded
        ndigits: amount of digits which needs to be rounded to
        
    Output:
        Rounded-up digit with ndigits digits.
    '''
    return int(mt.ceil(x / 10.0**ndigits)) * 10**ndigits

def plot(plotnumber):
    '''
    Reading data, processing and plotting of Figures 3a, 3b, 3c
    
    Input:
        plotnumber: Figure number from the article, input as a string
    '''
    subsets = 200
    roisize,m,_,_,_,other = vti.imgparams();
    delta_x = other[0]

    #################################################################################    
    
    if plotnumber=="3a":
        Jinv, thetaMAP, smp  = readdata(106, subsets) 
        maxval = np.max(smp[0])
        fig, ax = plt.subplots(ncols=3, nrows = 4, sharex = True, sharey = True)
        for k in range(3):
            for pat in range(4):
                ax[pat,k].imshow(smp[0,4*k+pat], cmap = 'inferno', vmin=0, vmax=maxval)
        
        ax[0,0].get_xaxis().set_ticks([])
        ax[0,0].get_yaxis().set_ticks([])  
        
        ax[0,0].set_ylabel(r'$\phi_{x,k}^+$', rotation = 'horizontal', ha = 'right', va = 'center', labelpad = 5)
        ax[1,0].set_ylabel(r'$\phi_{x,k}^-$', rotation = 'horizontal', ha = 'right', va = 'center', labelpad = 5)
        ax[2,0].set_ylabel(r'$\phi_{y,k}^+$', rotation = 'horizontal', ha = 'right', va = 'center', labelpad = 5)
        ax[3,0].set_ylabel(r'$\phi_{y,k}^-$', rotation = 'horizontal', ha = 'right', va = 'center', labelpad = 5)
        
        for k in range(3):
            itno = str(k+1)
            label = 'Iteration ' + itno + ' of 3'
            ax[0,k].set_title(label, fontsize = 12)
    
        fig.set_size_inches(5.16, 5, forward=True)
        plt.tight_layout()  
    
    #################################################################################    
    
    if plotnumber=="3b":
        Jinv, thetaMAP, smp  = readdata(3, subsets)
        
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True)
        
        pl1 = ax1.hist2d((thetaMAP[:,0,0]-5.5)*delta_x*10**9, (thetaMAP[:,0,1]-5.5)*delta_x*10**9, bins = 100, range=[[-0.25*delta_x*10**9,0.25*delta_x*10**9],[-0.25*delta_x*10**9,0.25*delta_x*10**9]])
        pl2 = ax2.hist2d((thetaMAP[:,1,0]-5.5)*delta_x*10**9, (thetaMAP[:,1,1]-5.5)*delta_x*10**9, bins = 100, range=[[-0.25*delta_x*10**9,0.25*delta_x*10**9],[-0.25*delta_x*10**9,0.25*delta_x*10**9]])
        pl3 = ax3.hist2d((thetaMAP[:,2,0]-5.5)*delta_x*10**9, (thetaMAP[:,2,1]-5.5)*delta_x*10**9, bins = 100, range=[[-0.25*delta_x*10**9,0.25*delta_x*10**9],[-0.25*delta_x*10**9,0.25*delta_x*10**9]])
        
        ax3.set_xlim(left=-0.25*delta_x*10**9, right = 0.25*delta_x*10**9)
        ax3.set_ylim(bottom=-0.25*delta_x*10**9, top = 0.25*delta_x*10**9)
        
        ax1.set_title('Iteration 1 of 3', fontsize = 12)
        ax2.set_title('Iteration 2 of 3', fontsize = 12)
        ax3.set_title('Iteration 3 of 3', fontsize = 12)
        ax1.set_xlabel('x-location [nm]')
        ax1.set_ylabel('y-location [nm]')
        ax2.set_xlabel('x-location [nm]')
        ax3.set_xlabel('x-location [nm]')
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        
        fig.colorbar(pl1[3], ax=ax1, fraction=0.046, pad=0.04)
        fig.colorbar(pl2[3], ax=ax2, fraction=0.046, pad=0.04)
        fig.colorbar(pl3[3], ax=ax3, fraction=0.046, pad=0.04)
        
        fig.set_size_inches(7.75, 2.75, forward=True)    

    #################################################################################    
        
    if plotnumber=="3c":
        Jinv, thetaMAP, smp  = readdata(3, subsets)
        fig, axes = plt.subplots(ncols=3, sharex=True, sharey=False)
        binsizes = np.array([50,50,50])
        for k in range(3):
            axes[k].hist((thetaMAP[:,k,0]-5.5)*delta_x*10**9, bins=binsizes[k], range=[-0.25*delta_x*10**9,0.25*delta_x*10**9])
        
            # Make figure that shows histograms
            [counts, edges] = np.histogram((thetaMAP[:,k,0]-5.5)*delta_x*10**9, bins=binsizes[k], range=[-0.25*delta_x*10**9,0.25*delta_x*10**9])
            centers = np.zeros(counts.shape[0])
            for q in range(edges.shape[0] - 1):
                centers[q] = (edges[q] + edges[q + 1]) / 2
            
            # Calculate initial estimates of sigma and kappa
            sigma0 = np.std((thetaMAP[:,k,0]-5.5)*delta_x*10**9)
            kappa0 = np.max(counts)
            x0 = np.array([kappa0, sigma0])
            
            # Fit the Gaussian through the histogram data and save into array
            p_opt = least_squares(error, 
                                  x0, 
                                  jac = jac, 
                                  args = (centers, counts));
            
            #Plot the Gaussian on top of the histogram
            xg = np.linspace(-1.2*delta_x*10**9,1.2*delta_x*10**9, 1200)
            yg = model(p_opt.x, xg)
            
            #Add the Gaussian fits to the histograms
            axes[k].plot(xg,yg)
            axes[k].text(-10, 1.05*max(counts), '${\sigma}_x$ = %.1f nm' % p_opt.x[1], fontsize = 13)
            string = 'Iteration ' + str(k+1) + ' of 3'
            axes[k].set_title(string, fontsize = 12)
            axes[k].set_xlabel('x-location [nm]')
            axes[k].set_xlim(left=-0.25*delta_x*10**9, right = 0.25*delta_x*10**9)
            axes[k].set_ylim(top = roundup(1.2*max(counts),ndigits=3))
            axes[k].set_aspect((0.5*delta_x*10**9)/roundup(1.2*max(counts),ndigits=3))
        
        axes[0].set_ylabel('Counts')
        fig.set_size_inches(7.5, 2.15, forward=True)

#Figure 3a
plot("3a")

#Figure 3b
plot("3b")

#Figure 3c
plot("3c")