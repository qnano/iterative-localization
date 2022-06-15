# -*- coding: utf-8 -*-
"""
Reading data, processing and plotting of Figures
    2d, 2e, 
    S2a, S2b, 
    S5, 
    S6a, S6b, S6c, 
    S7a, S7b, 
    S8a, S8b, S8c, 
    S10a, S10b, 
    S11a, S11b
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import VTI_helper as vti
import matplotlib as mpl

new_rc_params = {
"font.size": 12,
"axes.labelsize": 12}
mpl.rcParams.update(new_rc_params)

def RMSE(estimate,true):
    '''
    RMSE returns the root mean square error given:
        
    Input:
        estimate: a vector containing estimates of the parameter
        true: true parameter value (scalar)
        
    Output:
        RMSE: root mean square error of estimates
    '''
    
    deviation = estimate-true;
    MSE = np.mean(deviation**2)
    RMSE = np.sqrt(MSE)
    return RMSE

def readdata(setnumber, subsets):
    '''
    Reads VTI and MAP data from ./SimData/
    
    Input:
        setnumber: number of dataset as given in Datasets.xlsx
        subsets: amount of subsets used for RMSE computation
        
    Output:
        params: Imaging parameters for dataset
        CRLBx: CRLB/VTI for dataset
        thetaMAP: stack of MAP estimates for dataset
        RMSE: RMSE of thetaMAP
        RMSE_stdev: stdev of RMSE of thetaMAP
    '''
    loadpath = './SimData/'

    datasets_pd = pd.read_excel('Datasets.xlsx')
    datasets = np.array(datasets_pd)
    
    roisize,m,_,_,_,other = vti.imgparams();
    delta_x = other[0]
    
    # Excel numbering starts at 1
    setnumber -= 1
    
    itermax = int(datasets[setnumber,1]);
    alpha = datasets[setnumber,2];
    thetaI = datasets[setnumber,3];
    m = datasets[setnumber,4];
    thetab = datasets[setnumber,5];
    params = [itermax,alpha,thetaI,m,thetab]
    
    saveCRLB = 'VTIx-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    savethetaMAP = 'thetaMAP-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m))+ '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    savetheta = 'theta-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m))+ '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'

    thetaMAP = np.load(loadpath+savethetaMAP)
    CRLBx = np.load(loadpath+saveCRLB)
    theta_per_spot = np.load(loadpath+savetheta)
    
    RMSE_mat = np.zeros((subsets,itermax));

    for iteration in range(itermax):
        for subset in range(subsets):
            RMSE_mat[subset,iteration]=RMSE(thetaMAP[subset*len(thetaMAP)//subsets:(subset+1)*len(thetaMAP)//subsets,iteration,0],theta_per_spot[subset*len(thetaMAP)//subsets:(subset+1)*len(thetaMAP)//subsets,0])*delta_x 

    RMSE_val = np.average(RMSE_mat,axis=0)
    RMSE_stdev = np.std(RMSE_mat,axis=0)
    
    return params,CRLBx,RMSE_val,RMSE_stdev

def readdataT(setnumber):
    '''
    Reads data of analytical approximation of the VTI and quadratic approximation of the CRLB from ./SimData/
    
    Input:
        setnumber: number of dataset as given in Datasets.xlsx
        
    Output:
        CRLBxt: Analytical approximation of the VTI for dataset
        CRLBxM: Quadratic approximation of the CRLB for dataset
    '''
    loadpath = './SimData/'

    datasets_pd = pd.read_excel('Datasets.xlsx')
    datasets = np.array(datasets_pd)
    
    # Excel numbering starts at 1
    setnumber -= 1
    
    itermax = int(datasets[setnumber,1]);
    alpha = datasets[setnumber,2];
    thetaI = datasets[setnumber,3];
    thetab = datasets[setnumber,5];
    
    saveCRLBt = 'VTIxT-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(0))+ '.npy'
    CRLBxt = np.load(loadpath+saveCRLBt)
    
    saveCRLBM = 'CRLBxM-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    CRLBxM = np.load(loadpath+saveCRLBM)
    
    return CRLBxt,CRLBxM

def readdataI(setnumber):
    '''
    Reads data of the CRLB over all iterations from ./SimData/
    
    Input:
        setnumber: number of dataset as given in Datasets.xlsx
        
    Output:
        CRLBxI: Analytical approximation of the VTI for dataset
    '''
    loadpath = './SimData/'

    datasets_pd = pd.read_excel('Datasets.xlsx')
    datasets = np.array(datasets_pd)
    
    # Excel numbering starts at 1
    setnumber -= 1
    
    itermax = int(datasets[setnumber,1]);
    alpha = datasets[setnumber,2];
    thetaI = datasets[setnumber,3];
    m = datasets[setnumber,4];
    thetab = datasets[setnumber,5];
    
    saveCRLBI = 'CRLBxI-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    CRLBxI = np.load(loadpath+saveCRLBI)
    
    return CRLBxI

def readdata_thetaI(setnumbers, subsets, iters, stepsmax=120, readalpha=False):
    '''
    Reads VTI and MAP data from ./SimData/, for a range of datasets with different thetaI values.
    
    Input:
        setnumbers: list of dataset numbers as given in Datasets.xlsx
        subsets: amount of subsets used for RMSE computation
        iters: Amount of iterations corresponding to the dataset.
        
    Optional input:
        stepsmax: amount of intermediate steps used in the datasets
        readalpha: report values of the aggressiveness parameter for the different datasets
        
    Output:
        xrange: Range of thetaI values
        CRLBx: CRLB/VTI for datasets
        RMSE: RMSE of thetaMAP
        RMSE_stdev: stdev of RMSE of thetaMAP        
        
    Optional output:
        alpharange: Range of values of the used aggressiveness parameters
        CRLBx_1i: CRLB for datasets when 1 iteration is used, on the aggressiveness parameters of alpharange.
    '''
    params_mat = np.empty((len(setnumbers),5))
    CRLBx_mat = np.empty((len(setnumbers),stepsmax))
    RMSE_mat = np.empty((len(setnumbers),iters))
    RMSEstdev_mat = np.empty((len(setnumbers),iters))
    
    for i in range(len(setnumbers)):
        params, CRLBx, RMSE,  RMSEstdev = readdata(setnumbers[i], subsets)
        params_mat[i,:] = params
        CRLBx_mat[i,:] = CRLBx
        RMSE_mat[i,:] = RMSE
        RMSEstdev_mat[i,:] = RMSEstdev
        
    xrange = params_mat[:,2]
    CRLBx = CRLBx_mat[:,-1]
    RMSE = RMSE_mat[:,-1]
    RMSEstdev = RMSEstdev_mat[:,-1]
    
    if readalpha==False:
        return xrange, CRLBx, RMSE, RMSEstdev
    else:
        alpharange = params_mat[:,1]
        CRLBx_1i = CRLBx_mat[0,int(stepsmax/iters-1)]
        return xrange, CRLBx, RMSE, RMSEstdev, alpharange, CRLBx_1i

def readdataT_thetaI(setnumbers, stepsmax=120):
    '''
    Reads data of analytical approximation of the VTI and quadratic approximation of the CRLB from ./SimData/, 
    for a range of datasets with different thetaI values.
    
    Input:
        setnumbers: list of dataset numbers as given in Datasets.xlsx
        
    Optional input:
        stepsmax: amount of intermediate steps used in the datasets
           
    Output:
        CRLBxt: Analytical approximation of the VTI for datasets
        CRLBxM: Quadratic approximation of the CRLB for datasets
    '''
    CRLBxt_mat = np.empty((len(setnumbers),stepsmax))
    CRLBxM_mat = np.empty((len(setnumbers),stepsmax))
    
    for i in range(len(setnumbers)):
        CRLBxt, CRLBxM = readdataT(setnumbers[i])
        CRLBxt_mat[i,:] = CRLBxt
        CRLBxM_mat[i,:] = CRLBxM
        
    CRLBxt = CRLBxt_mat[:,-1]
    CRLBxM = CRLBxM_mat[:,-1]
    return CRLBxt, CRLBxM

def readdataI_thetaI(setnumbers, stepsmax=120):
    '''
    Reads data of the CRLB over all iterations from ./SimData/, 
    for a range of datasets with different thetaI values.
    
    Input:
        setnumbers: list of dataset numbers as given in Datasets.xlsx

    Optional input:
        stepsmax: amount of intermediate steps used in the datasets
            
    Output:
        CRLBxI: Analytical approximation of the VTI for datasets
    '''
    CRLBxI_mat = np.empty((len(setnumbers),stepsmax))
    
    for i in range(len(setnumbers)):
        CRLBxI = readdataI(setnumbers[i])
        CRLBxI_mat[i,:] = CRLBxI
        
    CRLBxI = CRLBxI_mat[:,-1]
    return CRLBxI
        
def plot(plotnumber, panel):
    '''
    Reading data, processing and plotting of Figures 2d, 2e, S2a, S2b, S5, S6a, S6b, S6c, 
    S7a, S7b, S8a, S8b, S8c, S10a, S10b, S11a, S11b
    
    Input:
        plotnumber: Figure number from the article, input as a string
        panel: Panel number. Use 1 for the left panel, 2 for the right panel and 0 for the legend.
    '''
    
    subsets = 200
    stepsmax = 120
    thetaI = 2000;

    #################################################################################

    if plotnumber == "2d":
        # 1 iteration
        setnumbers_1i = [37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
        xrange_1i, CRLBx_1i, RMSE_1i, RMSEstdev_1i = readdata_thetaI(setnumbers_1i, subsets, iters=1)

        # 2 iterations
        setnumbers_2i = [38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80]
        xrange_2i, CRLBx_2i, RMSE_2i, RMSEstdev_2i = readdata_thetaI(setnumbers_2i, subsets, iters=2)
        
        # 3 iterations  
        setnumbers_3i = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
        xrange_3i, CRLBx_3i, RMSE_3i, RMSEstdev_3i = readdata_thetaI(setnumbers_3i, subsets, iters=3)
        
        #VTI Theoretical
        #1 iteration
        CRLBxt_1i, CRLBxM_1i = readdataT_thetaI(setnumbers_1i)
        
        #2 iterations
        CRLBxt_2i, CRLBxM_2i = readdataT_thetaI(setnumbers_2i)
        
        #3 iterations
        CRLBxt_3i, CRLBxM_3i = readdataT_thetaI(setnumbers_3i)
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.loglog(xrange_1i,CRLBx_1i*10**9,color='C0')
            ax.loglog(xrange_2i,CRLBx_2i*10**9,color='C1')
            ax.loglog(xrange_3i,CRLBx_3i*10**9,color='C2')
    
            #MAP
            ax.errorbar(xrange_1i, RMSE_1i*10**9, yerr=RMSEstdev_1i*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(xrange_2i, RMSE_2i*10**9, yerr=RMSEstdev_2i*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_3i, RMSE_3i*10**9, yerr=RMSEstdev_3i*10**9, capsize=3, color='C2', marker='x', linestyle='None')
    
            #VTI Theoretical
            ax.loglog(xrange_1i,CRLBxt_1i*10**9,color='C0', linestyle = 'dashed')
            ax.loglog(xrange_2i,CRLBxt_2i*10**9,color='C1', linestyle = 'dashed')
            ax.loglog(xrange_3i,CRLBxt_3i*10**9,color='C2', linestyle = 'dashed')
            
            #MINFLUX
            ax.loglog(xrange_1i,CRLBxM_1i*10**9,color='C0', linestyle = 'dotted')
            ax.loglog(xrange_2i,CRLBxM_2i*10**9,color='C1', linestyle = 'dotted')
            ax.loglog(xrange_3i,CRLBxM_3i*10**9,color='C2', linestyle = 'dotted')
            
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom = 1*10**-4, top = 30)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange_2i,CRLBx_1i/CRLBx_2i,color='C1')
            ax.loglog(xrange_3i,CRLBx_1i/CRLBx_3i,color='C2')
    
            #MAP
            ax.errorbar(xrange_2i, CRLBx_1i/RMSE_2i, yerr=np.abs(CRLBx_1i/RMSE_2i - CRLBx_1i/(RMSE_2i+RMSEstdev_2i)), capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_3i, CRLBx_1i/RMSE_3i, yerr=np.abs(CRLBx_1i/RMSE_3i - CRLBx_1i/(RMSE_3i+RMSEstdev_3i)), capsize=3, color='C2', marker='x', linestyle='None')
    
            #VTI Theoretical
            ax.loglog(xrange_2i,CRLBxt_1i/CRLBxt_2i,color='C1', linestyle = 'dashed')
            ax.loglog(xrange_3i,CRLBxt_1i/CRLBxt_3i,color='C2', linestyle = 'dashed')
       
            #MINFLUX
            ax.loglog(xrange_2i,CRLBxM_1i/CRLBxM_2i,color='C1', linestyle = 'dotted')
            ax.loglog(xrange_3i,CRLBxM_1i/CRLBxM_3i,color='C2', linestyle = 'dotted')      
       
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom=7*10**-1, top = 1*10**4)        
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label=('RMSE of MAP estimates' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'), marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'2 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
    
            ax.legend(handles=[black_cont, black_VTI, black_M, black_dash, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

    #################################################################################
    
    if plotnumber == "2e":
        roisize,m,kx,ky,sigmap,other = vti.imgparams()
        omega = other[2]
        
        # MC-VTI 2 it., thetaI= 500
        setnumbers_500 = [448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462]
        xrange_500, CRLBx_500, RMSE_500, RMSEstdev_500, alpharange_500, CRLBx_500_1i = readdata_thetaI(setnumbers_500, subsets, iters=2, readalpha=True)
        phirange_500 = 2*omega*alpharange_500*CRLBx_500_1i

        # MC-VTI 2 it., thetaI = 1000
        setnumbers_1000 = [463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
        xrange_1000, CRLBx_1000, RMSE_1000, RMSEstdev_1000, alpharange_1000, CRLBx_1000_1i = readdata_thetaI(setnumbers_1000, subsets, iters=2, readalpha=True)
        phirange_1000 = 2*omega*alpharange_1000*CRLBx_1000_1i
        
        # MC-VTI 2 it., thetaI = 2000
        setnumbers_2000 = [319, 324, 329, 334, 339, 344, 349, 350, 351, 352, 353, 354, 355, 356, 357]
        xrange_2000, CRLBx_2000, RMSE_2000, RMSEstdev_2000, alpharange_2000, CRLBx_2000_1i = readdata_thetaI(setnumbers_2000, subsets, iters=2, readalpha=True)
        phirange_2000 = 2*omega*alpharange_2000*CRLBx_2000_1i
        
        # MC-VTI 2 it., thetaI = 5000
        setnumbers_5000 = [478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492]
        xrange_5000, CRLBx_5000, RMSE_5000, RMSEstdev_5000, alpharange_5000, CRLBx_5000_1i = readdata_thetaI(setnumbers_5000, subsets, iters=2, readalpha=True)
        phirange_5000 = 2*omega*alpharange_5000*CRLBx_5000_1i
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.semilogy(alpharange_500,CRLBx_500*10**9,color='C0')
            ax.semilogy(alpharange_1000,CRLBx_1000*10**9,color='C1')
            ax.semilogy(alpharange_2000,CRLBx_2000*10**9,color='C2')
            ax.semilogy(alpharange_5000,CRLBx_5000*10**9,color='C3')
                
            #MAP
            ax.errorbar(alpharange_500, RMSE_500*10**9, yerr=RMSEstdev_500*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(alpharange_1000, RMSE_1000*10**9, yerr=RMSEstdev_1000*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(alpharange_2000, RMSE_2000*10**9, yerr=RMSEstdev_2000*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(alpharange_5000, RMSE_5000*10**9, yerr=RMSEstdev_5000*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #Configuration
            ax.set_xlabel(r'Aggressiveness parameter $\alpha$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_ylim(bottom=10**-1)  
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        elif panel==2:
            #VTI
            ax.semilogy(phirange_500,CRLBx_500*10**9,color='C0')
            ax.semilogy(phirange_1000,CRLBx_1000*10**9,color='C1')
            ax.semilogy(phirange_2000,CRLBx_2000*10**9,color='C2')
            ax.semilogy(phirange_5000,CRLBx_5000*10**9,color='C3')
            
            #MAP
            ax.errorbar(phirange_500, RMSE_500*10**9, yerr=RMSEstdev_500*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(phirange_1000, RMSE_1000*10**9, yerr=RMSEstdev_1000*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(phirange_2000, RMSE_2000*10**9, yerr=RMSEstdev_2000*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(phirange_5000, RMSE_5000*10**9, yerr=RMSEstdev_5000*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #Configuration
            ax.set_xlabel(r'x-phase between pattern minima $\phi_{x,2}^+-\phi_{x,2}^-$ [rad]')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()  
            ax.set_ylim(bottom=10**-1)  
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        elif panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'(2 iterations, $m=0.95$,' + '\n' + r'$\theta_b$=8 photons/pixel)'))
            black_dash = mlines.Line2D([0], [0],color='black', label=('RMSE of MAP estimates'+ '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'), marker='x', linestyle='None')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$\theta_I = 500$ photons')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$\theta_I = 1000$ photons')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$\theta_I = 2000$ photons')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'$\theta_I = 5000$ photons')
     
            ax.legend(handles=[black_cont, black_dash, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

    #################################################################################

    if plotnumber == "S2a":
        # 1 iteration
        setnumbers_1i = [37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
        xrange_1i, CRLBx_1i, RMSE_1i, RMSEstdev_1i = readdata_thetaI(setnumbers_1i, subsets, iters=1)

        # 2 iterations
        setnumbers_2i = [38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80]
        xrange_2i, CRLBx_2i, RMSE_2i, RMSEstdev_2i = readdata_thetaI(setnumbers_2i, subsets, iters=2)
        
        # 3 iterations  
        setnumbers_3i = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
        xrange_3i, CRLBx_3i, RMSE_3i, RMSEstdev_3i = readdata_thetaI(setnumbers_3i, subsets, iters=3)

        # 5 iterations
        setnumbers_5i = [107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]
        xrange_5i, CRLBx_5i, RMSE_5i, RMSEstdev_5i = readdata_thetaI(setnumbers_5i, subsets, iters=5)
        
        #VTI Theoretical
        #1 iteration
        CRLBxI_1i = readdataI_thetaI(setnumbers_1i)
        
        #2 iterations
        CRLBxI_2i = readdataI_thetaI(setnumbers_2i)
        
        #3 iterations
        CRLBxI_3i = readdataI_thetaI(setnumbers_3i)
        
        #5 iterations
        CRLBxI_5i = readdataI_thetaI(setnumbers_5i)
         
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.loglog(xrange_1i,CRLBx_1i*10**9,color='C0')
            ax.loglog(xrange_2i,CRLBx_2i*10**9,color='C1')
            ax.loglog(xrange_3i,CRLBx_3i*10**9,color='C2')
            ax.loglog(xrange_5i,CRLBx_5i*10**9,color='C3')
            
            #CRLB
            ax.loglog(xrange_1i,CRLBxI_1i*10**9,color='C0', linestyle = 'dashed')
            ax.loglog(xrange_2i,CRLBxI_2i*10**9,color='C1', linestyle = 'dashed')
            ax.loglog(xrange_3i,CRLBxI_3i*10**9,color='C2', linestyle = 'dashed')
            ax.loglog(xrange_5i,CRLBxI_5i*10**9,color='C3', linestyle = 'dashed')
            
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=300, right = 10000)
            ax.set_ylim(bottom = 1*10**-1, top = 30)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange_2i,CRLBx_1i/CRLBx_2i,color='C1')
            ax.loglog(xrange_3i,CRLBx_1i/CRLBx_3i,color='C2')
            ax.loglog(xrange_5i,CRLBx_1i/CRLBx_5i,color='C3')
       
            #MINFLUX
            ax.loglog(xrange_2i,CRLBxI_1i/CRLBxI_2i,color='C1', linestyle = 'dashed')
            ax.loglog(xrange_3i,CRLBxI_1i/CRLBxI_3i,color='C2', linestyle = 'dashed') 
            ax.loglog(xrange_5i,CRLBxI_1i/CRLBxI_5i,color='C3', linestyle = 'dashed')  
       
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=300, right = 10000)
            ax.set_ylim(bottom=7*10**-1, top = 1*10**1)        
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB, computed over' + '\n' + 'all configurations' + '\n' +r'($m=0.95$, $\theta_b = 8$ ph./px.)'), linestyle='dashed')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'2 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'5 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
    
            ax.legend(handles=[black_cont, black_M, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)    



    #################################################################################
    
    if plotnumber == "S2b":
        alpha01 = 0.2
        alpha02 = 0.3
        alpha1 = 0.4
        alpha2 = 2/3
        alpha3 = 1
        
        # Uses datasets 1, 2, 3
        params1, CRLBx1, RMSE1,  RMSEstdev1 = readdata(1, subsets)
        params2, CRLBx2, RMSE2,  RMSEstdev2 = readdata(2, subsets)
        params3, CRLBx3, RMSE3,  RMSEstdev3 = readdata(3, subsets)
        params31, CRLBx31, RMSE31,  RMSEstdev31 = readdata(31, subsets)
        params32, CRLBx32, RMSE32,  RMSEstdev32 = readdata(32, subsets)
        params33, CRLBx33, RMSE33,  RMSEstdev33 = readdata(33, subsets)
        
        CRLBxI1 = readdataI(1)
        CRLBxI2 = readdataI(2)
        CRLBxI3 = readdataI(3)
        CRLBxI31 = readdataI(31)
        
        xrange1 = np.linspace(params1[2]/(stepsmax), params1[2], stepsmax)
        xrange2 = np.linspace(params2[2]/(stepsmax), params2[2], stepsmax)
        xrange3 = np.linspace(params3[2]/(stepsmax), params3[2], stepsmax) 
        xrange31 = np.linspace(params31[2]/(stepsmax), params31[2], stepsmax) 
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.loglog(xrange1, CRLBx1*10**9,color='C0', alpha = alpha3, marker = '^', markevery=[-1])
            
            ax.loglog(xrange2[0:60], CRLBx2[0:60]*10**9,color='C1', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1], CRLBx2[59:-1]*10**9,color='C1', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40], CRLBx3[0:40]*10**9,color='C2', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80], CRLBx3[39:80]*10**9,color='C2', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1], CRLBx3[79:-1]*10**9,color='C2', alpha = alpha3, marker = 'd', markevery=[-1])
    
            ax.loglog(xrange31[0:24], CRLBx31[0:24]*10**9,color='C3', alpha = alpha01, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[23:48], CRLBx31[23:48]*10**9,color='C3', alpha = alpha02, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[47:72], CRLBx31[47:72]*10**9,color='C3', alpha = alpha1, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[71:96], CRLBx31[71:96]*10**9,color='C3', alpha = alpha2, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[95:-1], CRLBx31[95:-1]*10**9,color='C3', alpha = alpha3, marker = 'p', markevery=[-1])
            
            #VTI Theoretical
            ax.loglog(xrange1, CRLBxI1*10**9,color='C0', alpha = alpha3, linestyle = 'dashed', marker = '^', markevery=[-1])
            
            ax.loglog(xrange2[0:60], CRLBxI2[0:60]*10**9,color='C1', alpha = alpha2, linestyle = 'dashed', marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1], CRLBxI2[59:-1]*10**9,color='C1', alpha = alpha3, linestyle = 'dashed', marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40], CRLBxI3[0:40]*10**9,color='C2', alpha = alpha1, linestyle = 'dashed', marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80], CRLBxI3[39:80]*10**9,color='C2', alpha = alpha2, linestyle = 'dashed', marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1], CRLBxI3[79:-1]*10**9,color='C2', alpha = alpha3, linestyle = 'dashed', marker = 'd', markevery=[-1])
    
            ax.loglog(xrange31[0:24], CRLBxI31[0:24]*10**9,color='C3', alpha = alpha01, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[23:48], CRLBxI31[23:48]*10**9,color='C3', alpha = alpha02, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[47:72], CRLBxI31[47:72]*10**9,color='C3', alpha = alpha1, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[71:96], CRLBxI31[71:96]*10**9,color='C3', alpha = alpha2, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[95:-1], CRLBxI31[95:-1]*10**9,color='C3', alpha = alpha3, linestyle = 'dashed', marker = 'p', markevery=[-1])
            
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=thetaI/10, right = thetaI+100)
            ax.set_ylim(bottom = 3*10**-1, top = 20)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange2[0:60],CRLBx1[0:60]/CRLBx2[0:60], color='C1', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1],CRLBx1[59:-1]/CRLBx2[59:-1], color='C1', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBx1[0:40]/CRLBx3[0:40], color='C2', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBx1[39:80]/CRLBx3[39:80], color='C2', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBx1[79:-1]/CRLBx3[79:-1], color='C2', alpha = alpha3, marker = 'd', markevery=[-1])
    
            ax.loglog(xrange31[0:24], CRLBx1[0:24]/CRLBx31[0:24],color='C3', alpha = alpha01, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[23:48], CRLBx1[23:48]/CRLBx31[23:48],color='C3', alpha = alpha02, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[47:72], CRLBx1[47:72]/CRLBx31[47:72],color='C3', alpha = alpha1, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[71:96], CRLBx1[71:96]/CRLBx31[71:96],color='C3', alpha = alpha2, marker = 'p', markevery=[-1])
            ax.loglog(xrange31[95:-1], CRLBx1[95:-1]/CRLBx31[95:-1],color='C3', alpha = alpha3, marker = 'p', markevery=[-1])
    
            #VTI Theoretical
            ax.loglog(xrange2[0:60],CRLBxI1[0:60]/CRLBxI2[0:60], color='C1', alpha = alpha2, linestyle = 'dashed', marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1],CRLBxI1[59:-1]/CRLBxI2[59:-1], color='C1', alpha = alpha3, linestyle = 'dashed', marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBxI1[0:40]/CRLBxI3[0:40], color='C2', alpha = alpha1, linestyle = 'dashed', marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBxI1[39:80]/CRLBxI3[39:80], color='C2', alpha = alpha2, linestyle = 'dashed', marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBxI1[79:-1]/CRLBxI3[79:-1], color='C2', alpha = alpha3, linestyle = 'dashed', marker = 'd', markevery=[-1])
    
            ax.loglog(xrange31[0:24], CRLBxI1[0:24]/CRLBxI31[0:24],color='C3', alpha = alpha01, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[23:48], CRLBxI1[23:48]/CRLBxI31[23:48],color='C3', alpha = alpha02, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[47:72], CRLBxI1[47:72]/CRLBxI31[47:72],color='C3', alpha = alpha1, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[71:96], CRLBxI1[71:96]/CRLBxI31[71:96],color='C3', alpha = alpha2, linestyle = 'dashed', marker = 'p', markevery=[-1])
            ax.loglog(xrange31[95:-1], CRLBxI1[95:-1]/CRLBxI31[95:-1],color='C3', alpha = alpha3, linestyle = 'dashed', marker = 'p', markevery=[-1])
            
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=thetaI/10, right = thetaI+100)
            ax.set_ylim(bottom=7*10**-1, top = 10**1)        
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_VTI = mlines.Line2D([0], [0],color='black', label=('CRLB, computed over' + '\n' + 'all configurations' + '\n' +r'($m=0.95$, $\theta_b = 8$ ph./px.)'), linestyle='dashed')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$',marker='^')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'2 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$',marker='o')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$',marker='d')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'5 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$',marker='p')
            
            ax.legend(handles=[black_cont, black_VTI, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

    #################################################################################
    
    if plotnumber == "S5":
        #Opacity scale settings
        alpha1 = 0.4
        alpha2 = 2/3
        alpha3 = 1
        
        # Uses datasets 1, 2, 3
        params1, CRLBx1, RMSE1,  RMSEstdev1 = readdata(1, subsets)
        params2, CRLBx2, RMSE2,  RMSEstdev2 = readdata(2, subsets)
        params3, CRLBx3, RMSE3,  RMSEstdev3 = readdata(3, subsets)
        
        CRLBxt1, CRLBxM1 = readdataT(1)
        CRLBxt2, CRLBxM2 = readdataT(2)
        CRLBxt3, CRLBxM3 = readdataT(3)
        
        xrange1 = np.linspace(params1[2]/(stepsmax), params1[2], stepsmax)
        xrange2 = np.linspace(params2[2]/(stepsmax), params2[2], stepsmax)
        xrange3 = np.linspace(params3[2]/(stepsmax), params3[2], stepsmax)

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)

        if panel==1:                              
            #VTI
            ax.loglog(xrange1, CRLBx1*10**9,color='C0', alpha = alpha3, marker = '^', markevery=[-1])
            
            ax.loglog(xrange2[0:60], CRLBx2[0:60]*10**9,color='C1', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1], CRLBx2[59:-1]*10**9,color='C1', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40], CRLBx3[0:40]*10**9,color='C2', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80], CRLBx3[39:80]*10**9,color='C2', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1], CRLBx3[79:-1]*10**9,color='C2', alpha = alpha3, marker = 'd', markevery=[-1])
    
            #VTI Theoretical
            ax.loglog(xrange1, CRLBxt1*10**9,color='C0', linestyle = 'dashed', alpha = alpha3, marker = '^', markevery=[-1])
            
            ax.loglog(xrange2[0:60], CRLBxt2[0:60]*10**9,color='C1', linestyle='dashed', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1], CRLBxt2[59:-1]*10**9,color='C1', linestyle='dashed', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40], CRLBxt3[0:40]*10**9,color='C2', linestyle='dashed', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80], CRLBxt3[39:80]*10**9,color='C2', linestyle='dashed', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1], CRLBxt3[79:-1]*10**9,color='C2', linestyle='dashed', alpha = alpha3, marker = 'd', markevery=[-1])
            
            #MINFLUX
            ax.loglog(xrange1, CRLBxM1*10**9,color='C0', linestyle = 'dotted', alpha = alpha3, marker = '^', markevery=[-1])
    
            ax.loglog(xrange2[0:60], CRLBxM2[0:60]*10**9,color='C1', linestyle='dotted', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1], CRLBxM2[59:-1]*10**9,color='C1', linestyle='dotted', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40], CRLBxM3[0:40]*10**9,color='C2', linestyle='dotted', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80], CRLBxM3[39:80]*10**9,color='C2', linestyle='dotted', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1], CRLBxM3[79:-1]*10**9,color='C2', linestyle='dotted', alpha = alpha3, marker = 'd', markevery=[-1])
            
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom = 1*10**-4, top = 30)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange2[0:60],CRLBx1[0:60]/CRLBx2[0:60], color='C1', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1],CRLBx1[59:-1]/CRLBx2[59:-1], color='C1', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBx1[0:40]/CRLBx3[0:40], color='C2', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBx1[39:80]/CRLBx3[39:80], color='C2', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBx1[79:-1]/CRLBx3[79:-1], color='C2', alpha = alpha3, marker = 'd', markevery=[-1])
    
            #VTI Theoretical
            ax.loglog(xrange2[0:60],CRLBxt1[0:60]/CRLBxt2[0:60], color='C1', linestyle='dashed', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1],CRLBxt1[59:-1]/CRLBxt2[59:-1], color='C1', linestyle='dashed', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBxt1[0:40]/CRLBxt3[0:40], color='C2', linestyle='dashed', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBxt1[39:80]/CRLBxt3[39:80], color='C2', linestyle='dashed', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBxt1[79:-1]/CRLBxt3[79:-1], color='C2', linestyle='dashed', alpha = alpha3, marker = 'd', markevery=[-1])
       
            #MINFLUX
            ax.loglog(xrange2[0:60],CRLBxM1[0:60]/CRLBxM2[0:60], color='C1', linestyle='dotted', alpha = alpha2, marker = 'o', markevery=[-1])
            ax.loglog(xrange2[59:-1],CRLBxM1[59:-1]/CRLBxM2[59:-1], color='C1', linestyle='dotted', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBxM1[0:40]/CRLBxM3[0:40], color='C2', linestyle='dotted', alpha = alpha1, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBxM1[39:80]/CRLBxM3[39:80], color='C2', linestyle='dotted', alpha = alpha2, marker = 'd', markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBxM1[79:-1]/CRLBxM3[79:-1], color='C2', linestyle='dotted', alpha = alpha3, marker = 'd', markevery=[-1])        
       
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom=7*10**-1, top = 10**4)        
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$', marker = '^')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'2 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$', marker = 'o')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$', marker = 'd')
    
            ax.legend(handles=[black_cont, black_VTI, black_M, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

    #################################################################################
    
    elif plotnumber == "S6a":
        # MC-VTI 3 it., m = 0.8
        setnumbers_8 = [271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285]
        xrange_8, CRLBx_8, RMSE_8, RMSEstdev_8 = readdata_thetaI(setnumbers_8, subsets, iters=3)
        
        # MC-VTI 3 it., m = 0.9
        setnumbers_9 = [286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300]
        xrange_9, CRLBx_9, RMSE_9, RMSEstdev_9 = readdata_thetaI(setnumbers_9, subsets, iters=3)

        # MC-VTI 3 it., m = 1
        setnumbers_10 = [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315]
        xrange_10, CRLBx_10, RMSE_10, RMSEstdev_10 = readdata_thetaI(setnumbers_10, subsets, iters=3)
        RMSE_10 = RMSE_10[0:12]
        RMSEstdev_10 = RMSEstdev_10[0:12]
        xrangeRMSE_10 = xrange_10[0:12]
        
        # MC-VTI 1 it., m = 0.8
        params_8i, CRLBx_8i, RMSE_8i, RMSEstdev_8i = readdata(316, subsets)
        xrange_8i = np.linspace(params_8i[2]/(stepsmax), params_8i[2], stepsmax)
        CRLBx_8i = np.interp(xrange_8, xrange_8i, CRLBx_8i)
        
        # MC-VTI 1 it., m = 0.9
        params_9i, CRLBx_9i, RMSE_9i, RMSEstdev_9i = readdata(317, subsets)
        xrange_9i = np.linspace(params_9i[2]/(stepsmax), params_9i[2], stepsmax)
        CRLBx_9i = np.interp(xrange_9, xrange_9i, CRLBx_9i)
        
        # MC-VTI 1 it., m = 1
        params_10i, CRLBx_10i, RMSE_10i, RMSEstdev_10i = readdata(318, subsets)
        xrange_10i = np.linspace(params_10i[2]/(stepsmax), params_10i[2], stepsmax)
        CRLBx_10i = np.interp(xrange_10, xrange_10i, CRLBx_10i)
        CRLBxRMSE_10i = CRLBx_10i[0:12]
        
        #VTI Theoretical & MINFLUX 3 it., m = 1
        CRLBxt_10, CRLBxM_10 = readdataT_thetaI(setnumbers_10)
        
        #VTI Theoretical & MINFLUX 1 it., m = 1
        CRLBxt_10i, CRLBxM_10i = readdataT(318)
        CRLBxt_10i = np.interp(xrange_9, xrange_9i, CRLBxt_10i)
        CRLBxM_10i = np.interp(xrange_9, xrange_9i, CRLBxM_10i)
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.loglog(xrange_8,CRLBx_8*10**9,color='C0')
            ax.loglog(xrange_9,CRLBx_9*10**9,color='C1')
            ax.loglog(xrange_10,CRLBx_10*10**9,color='C2')
                
            #MAP
            ax.errorbar(xrange_8, RMSE_8*10**9, yerr=RMSEstdev_8*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(xrange_9, RMSE_9*10**9, yerr=RMSEstdev_9*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrangeRMSE_10, RMSE_10*10**9, yerr=RMSEstdev_10*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            
            #VTI Theoretical
            ax.loglog(xrange_10,CRLBxt_10*10**9,color='C2', linestyle='dashed')
            
            #MINFLUX
            ax.loglog(xrange_10,CRLBxM_10*10**9,color='C2', linestyle='dotted')
            
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom = 1*10**-4, top = 30)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange_8, CRLBx_8i/CRLBx_8, color='C0')
            ax.loglog(xrange_9, CRLBx_9i/CRLBx_9, color='C1')
            ax.loglog(xrange_10, CRLBx_10i/CRLBx_10, color='C2')
            
            #MAP
            ax.errorbar(xrange_8, CRLBx_8i/RMSE_8, yerr=np.abs(CRLBx_8i/RMSE_8 - CRLBx_8i/(RMSE_8+RMSEstdev_8)), capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(xrange_9, CRLBx_9i/RMSE_9, yerr=np.abs(CRLBx_9i/RMSE_9 - CRLBx_9i/(RMSE_9+RMSEstdev_9)), capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrangeRMSE_10, CRLBxRMSE_10i/RMSE_10, yerr=[np.abs(CRLBxRMSE_10i/RMSE_10 - CRLBxRMSE_10i/(RMSE_10+RMSEstdev_10)), np.abs(CRLBxRMSE_10i/RMSE_10 - CRLBxRMSE_10i/(RMSE_10-RMSEstdev_10)+10000000*np.array([0,0,0,0,0,0,0,1,1,1,1,1]))], capsize=3, color='C2', marker='x', linestyle='None')
            
            #VTI Theoretical
            ax.loglog(xrange_10,CRLBxt_10i/CRLBxt_10,color='C2', linestyle = 'dashed')
       
            #MINFLUX
            ax.loglog(xrange_10,CRLBxM_10i/CRLBxM_10,color='C2', linestyle = 'dotted')  
        
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom=4*10**-1, top = 1*10**4)        
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'(3 iterations, $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label='RMSE of MAP estimates', marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'(3 iterations, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'(3 iterations, $\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$m=0.8$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$m=0.9$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$m=1.0$')
     
            ax.legend(handles=[black_cont, black_VTI, black_M , black_dash, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)


    #################################################################################

    if plotnumber == "S6b":
        # Opacity settings
        alpha1 = 0.4
        alpha2 = 2/3
        alpha3 = 1

        # Uses datasets 10, 11, 13, 16, 18, 20
        params10, CRLBx10, RMSE10,  RMSEstdev10 = readdata(10, subsets)
        params11, CRLBx11, RMSE11,  RMSEstdev11 = readdata(11, subsets)
        params13, CRLBx13, RMSE13,  RMSEstdev13 = readdata(13, subsets)
        params16, CRLBx16, RMSE16,  RMSEstdev16 = readdata(16, subsets)
        params18, CRLBx18, RMSE18,  RMSEstdev18 = readdata(18, subsets)
        params20, CRLBx20, RMSE20,  RMSEstdev20 = readdata(20, subsets)
        
        CRLBxt10, CRLBxM10 = readdataT(10)
        CRLBxt11, CRLBxM11 = readdataT(11)
        CRLBxt13, CRLBxM13 = readdataT(13)
        CRLBxt16, CRLBxM16 = readdataT(16)
        CRLBxt18, CRLBxM18 = readdataT(18)
        CRLBxt20, CRLBxM20 = readdataT(20)
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        xrange10 = np.linspace(params10[2]/(stepsmax), params10[2], stepsmax)
        xrange11 = np.linspace(params11[2]/(stepsmax), params11[2], stepsmax)
        xrange13 = np.linspace(params13[2]/(stepsmax), params13[2], stepsmax)
        
        if panel==1:
            #VTI
            ax.loglog(xrange13[0:40],CRLBx13[0:40]*10**9,color='C0', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange13[39:80],CRLBx13[39:80]*10**9,color='C0', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange13[79:-1],CRLBx13[79:-1]*10**9,color='C0', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange11[0:40],CRLBx11[0:40]*10**9,color='C1', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange11[39:80],CRLBx11[39:80]*10**9,color='C1', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange11[79:-1],CRLBx11[79:-1]*10**9,color='C1', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange10[0:40],CRLBx10[0:40]*10**9,color='C2', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange10[39:80],CRLBx10[39:80]*10**9,color='C2', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange10[79:-1],CRLBx10[79:-1]*10**9,color='C2', alpha=alpha3, marker='d',markevery=[-1])
            
            #VTI Theoretical
            ax.loglog(xrange10[0:40],CRLBxt10[0:40]*10**9,color='C2', alpha=alpha1, linestyle = 'dashed', marker='d',markevery=[-1])
            ax.loglog(xrange10[39:80],CRLBxt10[39:80]*10**9,color='C2', alpha=alpha2, linestyle = 'dashed', marker='d',markevery=[-1])
            ax.loglog(xrange10[79:-1],CRLBxt10[79:-1]*10**9,color='C2', alpha=alpha3, linestyle = 'dashed', marker='d',markevery=[-1])
    
            #MINFLUX
            ax.loglog(xrange10[0:40],CRLBxM10[0:40]*10**9,color='C2', alpha=alpha1, linestyle = 'dotted', marker='d',markevery=[-1])
            ax.loglog(xrange10[39:80],CRLBxM10[39:80]*10**9,color='C2', alpha=alpha2, linestyle = 'dotted', marker='d',markevery=[-1])
            ax.loglog(xrange10[79:-1],CRLBxM10[79:-1]*10**9,color='C2', alpha=alpha3, linestyle = 'dotted', marker='d',markevery=[-1])
                    
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom = 10**-3, top = 20)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange13[0:40],CRLBx20[0:40]/CRLBx13[0:40], color='C0', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange13[39:80],CRLBx20[39:80]/CRLBx13[39:80], color='C0', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange13[79:-1],CRLBx20[79:-1]/CRLBx13[79:-1], color='C0', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange11[0:40],CRLBx18[0:40]/CRLBx11[0:40], color='C1', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange11[39:80],CRLBx18[39:80]/CRLBx11[39:80], color='C1', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange11[79:-1],CRLBx18[79:-1]/CRLBx11[79:-1], color='C1', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange10[0:40],CRLBx16[0:40]/CRLBx10[0:40], color='C2', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange10[39:80],CRLBx16[39:80]/CRLBx10[39:80], color='C2', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange10[79:-1],CRLBx16[79:-1]/CRLBx10[79:-1], color='C2', alpha=alpha3, marker='d',markevery=[-1])
            
            #VTI
            ax.loglog(xrange10[0:40],CRLBxt16[0:40]/CRLBxt10[0:40], color='C2', linestyle = 'dashed', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange10[39:80],CRLBxt16[39:80]/CRLBxt10[39:80], color='C2', linestyle = 'dashed', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange10[79:-1],CRLBxt16[79:-1]/CRLBxt10[79:-1], color='C2', linestyle = 'dashed', alpha=alpha3, marker='d',markevery=[-1])
    
            #MINFLUX
            ax.loglog(xrange10[0:40],CRLBxM16[0:40]/CRLBxM10[0:40], color='C2', linestyle = 'dotted', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange10[39:80],CRLBxM16[39:80]/CRLBxM10[39:80], color='C2', linestyle = 'dotted', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange10[79:-1],CRLBxM16[79:-1]/CRLBxM10[79:-1], color='C2', linestyle = 'dotted', alpha=alpha3, marker='d',markevery=[-1])
                    
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom=7*10**-1, top = 10**3)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
            
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'(3 iterations, $\theta_b = 8$ ph./px.)'))
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'(3 iterations, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'(3 iterations, $\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$m=0.8$', marker='d')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$m=0.9$', marker='d')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$m=1.0$', marker='d')
        
            ax.legend(handles=[black_cont, black_VTI, black_M, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

    #################################################################################
    
    elif plotnumber == "S6c":
        params10, CRLBx10, RMSE10,  RMSEstdev10 = readdata(10, subsets)
        params11, CRLBx11, RMSE11,  RMSEstdev11 = readdata(11, subsets)
        params13, CRLBx13, RMSE13,  RMSEstdev13 = readdata(13, subsets)
        params16, CRLBx16, RMSE16,  RMSEstdev16 = readdata(16, subsets)
        params17, CRLBx17, RMSE17,  RMSEstdev17 = readdata(17, subsets)
        params18, CRLBx18, RMSE18,  RMSEstdev18 = readdata(18, subsets)
        params19, CRLBx19, RMSE19,  RMSEstdev19 = readdata(19, subsets)
        params20, CRLBx20, RMSE20,  RMSEstdev20 = readdata(20, subsets)
        params21, CRLBx21, RMSE21,  RMSEstdev21 = readdata(21, subsets)
        
        CRLBxt10, CRLBxM10 = readdataT(10)
        CRLBxt11, CRLBxM11 = readdataT(11)
        CRLBxt13, CRLBxM13 = readdataT(13)
        CRLBxt16, CRLBxM16 = readdataT(16)
        CRLBxt17, CRLBxM17 = readdataT(17)
        CRLBxt18, CRLBxM18 = readdataT(18)
        CRLBxt19, CRLBxM19 = readdataT(19)
        CRLBxt20, CRLBxM20 = readdataT(20)
        CRLBxt21, CRLBxM21 = readdataT(21)

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:    
            #VTI
            ax.semilogy([1,2,3],np.array([CRLBx20[-1]*10**9, CRLBx21[-1]*10**9, CRLBx13[-1]*10**9]),color='C0')
            ax.semilogy([1,2,3],np.array([CRLBx18[-1]*10**9, CRLBx19[-1]*10**9, CRLBx11[-1]*10**9]),color='C1')
            ax.plot([1,2,3],np.array([CRLBx16[-1]*10**9, CRLBx17[-1]*10**9, CRLBx10[-1]*10**9]),color='C2')
           
            #MAP
            ax.errorbar([1,2,3], np.array([RMSE20[-1]*10**9, RMSE21[-1]*10**9, RMSE13[-1]*10**9]), yerr=np.array([RMSEstdev20[-1]*10**9, RMSEstdev21[-1]*10**9, RMSEstdev13[-1]*10**9]), capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar([1,2,3], np.array([RMSE18[-1]*10**9, RMSE19[-1]*10**9, RMSE11[-1]*10**9]), yerr=np.array([RMSEstdev18[-1]*10**9, RMSEstdev19[-1]*10**9, RMSEstdev11[-1]*10**9]), capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar([1,2,3], np.array([RMSE16[-1]*10**9, RMSE17[-1]*10**9, RMSE10[-1]*10**9]), yerr=np.array([RMSEstdev16[-1]*10**9, RMSEstdev17[-1]*10**9, RMSEstdev10[-1]*10**9]), capsize=3, color='C2', marker='x', linestyle='None')
     
            #VTI Theoretical
            ax.semilogy([1,2,3],np.array([CRLBxt20[-1]*10**9, CRLBxt21[-1]*10**9, CRLBxt13[-1]*10**9]),color='C0', linestyle='dashed')
            ax.semilogy([1,2,3],np.array([CRLBxt18[-1]*10**9, CRLBxt19[-1]*10**9, CRLBxt11[-1]*10**9]),color='C1', linestyle='dashed')
            ax.semilogy([1,2,3],np.array([CRLBxt16[-1]*10**9, CRLBxt17[-1]*10**9, CRLBxt10[-1]*10**9]),color='C2', linestyle='dashed')       
    
            #MINFLUX
            ax.semilogy([1,2,3],np.array([CRLBxM16[-1]*10**9, CRLBxM17[-1]*10**9, CRLBxM10[-1]*10**9]),color='C2', linestyle='dotted')     
    
            #Configuration
            ax.set_xlabel(r'Amount of iterations')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=0.8, right = 3.2)
            ax.set_ylim(bottom = 10**-3, top = 20)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
            ax.set_xticks([1, 2, 3])
            ax.xaxis.set_ticklabels([1, 2, 3])       
        
        if panel==2:      
            #VTI
            ax.semilogy([1,2,3],CRLBx20[-1]/np.array([CRLBx20[-1], CRLBx21[-1], CRLBx13[-1]]), color='C0')
            ax.semilogy([1,2,3],CRLBx18[-1]/np.array([CRLBx18[-1], CRLBx19[-1], CRLBx11[-1]]), color='C1')
            ax.semilogy([1,2,3],CRLBx16[-1]/np.array([CRLBx16[-1], CRLBx17[-1], CRLBx10[-1]]), color='C2')
            
            #MAP
            ax.errorbar([1,2,3], CRLBx20[-1]/np.array([RMSE20[-1], RMSE21[-1], RMSE13[-1]]), yerr=np.abs(CRLBx20[-1]/np.array([RMSE20[-1], RMSE21[-1], RMSE13[-1]]) - CRLBx20[-1]/(np.array([RMSE20[-1], RMSE21[-1], RMSE13[-1]])+np.array([RMSEstdev20[-1], RMSEstdev21[-1], RMSEstdev13[-1]]))), capsize=3, marker='x', color='C0', linestyle='None')
            ax.errorbar([1,2,3], CRLBx18[-1]/np.array([RMSE18[-1], RMSE19[-1], RMSE11[-1]]), yerr=np.abs(CRLBx18[-1]/np.array([RMSE18[-1], RMSE19[-1], RMSE11[-1]]) - CRLBx18[-1]/(np.array([RMSE18[-1], RMSE19[-1], RMSE11[-1]])+np.array([RMSEstdev18[-1], RMSEstdev19[-1], RMSEstdev11[-1]]))), capsize=3, marker='x', color='C1', linestyle='None')
            ax.errorbar([1,2,3], CRLBx16[-1]/np.array([RMSE16[-1], RMSE17[-1], RMSE10[-1]]), yerr=[np.abs(CRLBx16[-1]/np.array([RMSE16[-1], RMSE17[-1], RMSE10[-1]]) - CRLBx16[-1]/(np.array([RMSE16[-1], RMSE17[-1], RMSE10[-1]])+np.array([RMSEstdev16[-1], RMSEstdev17[-1], RMSEstdev10[-1]]))), np.abs(CRLBx16[-1]/np.array([RMSE16[-1], RMSE17[-1], RMSE10[-1]]) - CRLBx16[-1]/(np.array([RMSE16[-1], RMSE17[-1], RMSE10[-1]])+np.array([RMSEstdev16[-1], RMSEstdev17[-1], -RMSE10[-1]+0.0000000000001])))], capsize=3, marker='x', color='C2', linestyle='None')
    
            #VTI Theoretical
            ax.semilogy([1,2,3],CRLBxt16[-1]/np.array([CRLBxt16[-1], CRLBxt17[-1], CRLBxt10[-1]]), color='C2', linestyle='dashed')        
           
            #MINFLUX
            ax.semilogy([1,2,3],CRLBxM16[-1]/np.array([CRLBxM16[-1], CRLBxM17[-1], CRLBxM10[-1]]), color='C2', linestyle='dotted')   
            
            #Configuration
            ax.set_xlabel(r'Amount of iterations')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=0.8, right = 3.2)
            ax.set_ylim(bottom=7*10**-1, top = 10**3)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
            ax.set_xticks([1, 2, 3])
            ax.xaxis.set_ticklabels([1, 2, 3])
            
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($\theta_I=2000$ ph., $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label='RMSE of MAP estimates', marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($\theta_I=2000$ ph., $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'($\theta_I=2000$ ph., $\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$m=0.8$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$m=0.9$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$m=1.0$')
    
            ax.legend(handles=[black_cont, black_VTI,  black_M, black_dash, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)


    #################################################################################

    if plotnumber == "S7a":
        # sigma vs theta_I for different iteration counts

        # MC-VTI 3 it., 1 ph./px.
        setnumbers_1 = [122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]
        xrange_1, CRLBx_1, RMSE_1, RMSEstdev_1 = readdata_thetaI(setnumbers_1, subsets, iters=3)

        # MC-VTI 3 it., 4 ph./px
        setnumbers_4 = [137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151]
        xrange_4, CRLBx_4, RMSE_4, RMSEstdev_4 = readdata_thetaI(setnumbers_4, subsets, iters=3)
        
        # MC-VTI 3 it., 8 ph./px. 
        setnumbers_8 = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
        xrange_8, CRLBx_8, RMSE_8, RMSEstdev_8 = readdata_thetaI(setnumbers_8, subsets, iters=3)

        # MC-VTI 3 it., 12 ph./px. 
        setnumbers_12 = [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166]
        xrange_12, CRLBx_12, RMSE_12, RMSEstdev_12 = readdata_thetaI(setnumbers_12, subsets, iters=3)
        
        # MC-VTI 1 it.
        params_1i, CRLBx_1i, RMSE_1i, RMSEstdev_1i = readdata(197, subsets)
        params_4i, CRLBx_4i, RMSE_4i, RMSEstdev_4i = readdata(198, subsets)
        params_8i, CRLBx_8i, RMSE_8i, RMSEstdev_8i = readdata(79, subsets)
        params_12i, CRLBx_12i, RMSE_12i, RMSEstdev_12i = readdata(199, subsets)
        
        xrange_1i = np.linspace(params_1i[2]/(stepsmax), params_1i[2], stepsmax)
        xrange_4i = np.linspace(params_4i[2]/(stepsmax), params_4i[2], stepsmax)
        xrange_8i = np.linspace(params_8i[2]/(stepsmax), params_8i[2], stepsmax)
        xrange_12i = np.linspace(params_12i[2]/(stepsmax), params_12i[2], stepsmax)
        
        CRLBx_1i = np.interp(xrange_1, xrange_1i, CRLBx_1i)
        CRLBx_4i = np.interp(xrange_4, xrange_4i, CRLBx_4i)
        CRLBx_8i = np.interp(xrange_8, xrange_8i, CRLBx_8i)
        CRLBx_12i = np.interp(xrange_12, xrange_12i, CRLBx_12i)
        
        #VTI Theoretical & MINFLUX 3 it., 0 ph./px.
        setnumbers_0 = [182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
        CRLBxt_0, CRLBxM_0 = readdataT_thetaI(setnumbers_0)
        
        #VTI Theoretical & MINFLUX 1 it., 0 ph./px.
        CRLBxt_0i, CRLBxM_0i = readdataT(202)
        CRLBxt_0i = np.interp(xrange_1, xrange_1i, CRLBxt_0i)
        CRLBxM_0i = np.interp(xrange_1, xrange_1i, CRLBxM_0i)
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.loglog(xrange_1,CRLBx_1*10**9,color='C0')
            ax.loglog(xrange_4,CRLBx_4*10**9,color='C1')
            ax.loglog(xrange_8,CRLBx_8*10**9,color='C2')
            ax.loglog(xrange_12,CRLBx_12*10**9,color='C3')
                
            #MAP
            ax.errorbar(xrange_1, RMSE_1*10**9, yerr=RMSEstdev_1*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(xrange_4, RMSE_4*10**9, yerr=RMSEstdev_4*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_8, RMSE_8*10**9, yerr=RMSEstdev_8*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(xrange_12, RMSE_12*10**9, yerr=RMSEstdev_12*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #VTI Theoretical
            ax.loglog(xrange_1,CRLBxt_0*10**9,color='C4',linestyle='dashed')
            
            #MINFLUX
            ax.loglog(xrange_1,CRLBxM_0*10**9,color='C4',linestyle='dotted')
            
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom = 1*10**-4, top = 30)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange_1, CRLBx_1i/CRLBx_1, color='C0')
            ax.loglog(xrange_4, CRLBx_4i/CRLBx_4, color='C1')
            ax.loglog(xrange_8, CRLBx_8i/CRLBx_8, color='C2')
            ax.loglog(xrange_12, CRLBx_12i/CRLBx_12, color='C3')
            
            #MAP
            ax.errorbar(xrange_1, CRLBx_1i/RMSE_1, yerr=np.abs(CRLBx_1i/RMSE_1 - CRLBx_1i/(RMSE_1+RMSEstdev_1)), capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(xrange_4, CRLBx_4i/RMSE_4, yerr=np.abs(CRLBx_4i/RMSE_4 - CRLBx_4i/(RMSE_4+RMSEstdev_4)), capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_8, CRLBx_8i/RMSE_8, yerr=np.abs(CRLBx_8i/RMSE_8 - CRLBx_8i/(RMSE_8+RMSEstdev_8)), capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(xrange_12, CRLBx_12i/RMSE_12, yerr=np.abs(CRLBx_12i/RMSE_12 - CRLBx_12i/(RMSE_12+RMSEstdev_12)), capsize=3, color='C3', marker='x', linestyle='None')
            
            
            #VTI Theoretical
            ax.loglog(xrange_1,CRLBxt_0i/CRLBxt_0,color='C4', linestyle = 'dashed')
       
            #MINFLUX
            ax.loglog(xrange_1,CRLBxM_0i/CRLBxM_0,color='C4', linestyle = 'dotted')     
            
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom=4*10**-1, top = 1*10**4)        
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'(3 iterations, $m=0.95$)'))
            black_dash = mlines.Line2D([0], [0],color='black', label='RMSE of MAP estimates', marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'(3 iterations, $m=1$)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'(3 iterations, $m=1$)'), linestyle='dotted')
            C4_line = mlines.Line2D([0],[0],color='C4', label=r'$\theta_b$=0 photons/pixel')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$\theta_b$=1 photon/pixel')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$\theta_b$=4 photons/pixel')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$\theta_b$=8 photons/pixel')
            C3_line = mlines.Line2D([0],[0],color='C3', label=r'$\theta_b$=12 photons/pixel')
     
            ax.legend(handles=[black_cont, black_VTI, black_M , black_dash, C4_line, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)


    #################################################################################
    
    elif plotnumber == "S7b":
        # Opacity settings
        alpha1 = 0.4
        alpha2 = 2/3
        alpha3 = 1
        
        # Uses datasets 1, 3, 22, 23, 24, (25), 26, 27, 28, (29)
        params1, CRLBx1, RMSE1,  RMSEstdev1 = readdata(1, subsets)
        params3, CRLBx3, RMSE3,  RMSEstdev3 = readdata(3, subsets)
        params22, CRLBx22, RMSE22,  RMSEstdev22 = readdata(22, subsets)
        params23, CRLBx23, RMSE23,  RMSEstdev23 = readdata(23, subsets)
        params24, CRLBx24, RMSE24,  RMSEstdev24 = readdata(24, subsets)
        params25, CRLBx25, RMSE25,  RMSEstdev25 = readdata(25, subsets)
        params26, CRLBx26, RMSE26,  RMSEstdev26 = readdata(26, subsets)
        params27, CRLBx27, RMSE27,  RMSEstdev27 = readdata(27, subsets)
        params28, CRLBx28, RMSE28,  RMSEstdev28 = readdata(28, subsets)
        params29, CRLBx29, RMSE29,  RMSEstdev29 = readdata(29, subsets)
        
        CRLBxt1, CRLBxM1 = readdataT(1)
        CRLBxt3, CRLBxM3 = readdataT(3)
        CRLBxt22, CRLBxM22 = readdataT(22)
        CRLBxt23, CRLBxM23 = readdataT(23)
        CRLBxt24, CRLBxM24 = readdataT(24)
        CRLBxt25, CRLBxM25 = readdataT(25)
        CRLBxt26, CRLBxM26 = readdataT(26)
        CRLBxt27, CRLBxM27 = readdataT(27)
        CRLBxt28, CRLBxM28 = readdataT(28)
        CRLBxt29, CRLBxM29 = readdataT(29)
        
        xrange22=np.linspace(params22[2]/(stepsmax), params22[2], stepsmax)
        xrange23=np.linspace(params23[2]/(stepsmax), params23[2], stepsmax)
        xrange3=np.linspace(params3[2]/(stepsmax), params3[2], stepsmax)
        xrange24=np.linspace(params24[2]/(stepsmax), params24[2], stepsmax)

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI  
            ax.loglog(xrange22[0:40],CRLBx22[0:40]*10**9,color='C0', alpha=alpha1,marker='d',markevery=[-1])
            ax.loglog(xrange22[39:80],CRLBx22[39:80]*10**9,color='C0', alpha=alpha2,marker='d',markevery=[-1])
            ax.loglog(xrange22[79:-1],CRLBx22[79:-1]*10**9,color='C0', alpha=alpha3,marker='d',markevery=[-1])
            
            ax.loglog(xrange23[0:40],CRLBx23[0:40]*10**9,color='C1', alpha=alpha1,marker='d',markevery=[-1])
            ax.loglog(xrange23[39:80],CRLBx23[39:80]*10**9,color='C1', alpha=alpha2,marker='d',markevery=[-1])
            ax.loglog(xrange23[79:-1],CRLBx23[79:-1]*10**9,color='C1', alpha=alpha3,marker='d',markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBx3[0:40]*10**9,color='C2', alpha=alpha1,marker='d',markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBx3[39:80]*10**9,color='C2', alpha=alpha2,marker='d',markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBx3[79:-1]*10**9,color='C2', alpha=alpha3,marker='d',markevery=[-1])
            
            ax.loglog(xrange24[0:40],CRLBx24[0:40]*10**9,color='C3', alpha=alpha1,marker='d',markevery=[-1])
            ax.loglog(xrange24[39:80],CRLBx24[39:80]*10**9,color='C3', alpha=alpha2,marker='d',markevery=[-1])
            ax.loglog(xrange24[79:-1],CRLBx24[79:-1]*10**9,color='C3', alpha=alpha3,marker='d',markevery=[-1])
    
            #VTI Theoretical
            ax.loglog(xrange22[0:40],CRLBxt22[0:40]*10**9,color='C4', alpha=alpha1,linestyle='dashed',marker='d',markevery=[-1])
            ax.loglog(xrange22[39:80],CRLBxt22[39:80]*10**9,color='C4', alpha=alpha2,linestyle='dashed',marker='d',markevery=[-1])
            ax.loglog(xrange22[79:-1],CRLBxt22[79:-1]*10**9,color='C4', alpha=alpha3,linestyle='dashed',marker='d',markevery=[-1])
    
            #MINFLUX
            ax.loglog(xrange22[0:40],CRLBxM22[0:40]*10**9,color='C4', alpha=alpha1,linestyle='dotted',marker='d',markevery=[-1])
            ax.loglog(xrange22[39:80],CRLBxM22[39:80]*10**9,color='C4', alpha=alpha2,linestyle='dotted',marker='d',markevery=[-1])
            ax.loglog(xrange22[79:-1],CRLBxM22[79:-1]*10**9,color='C4', alpha=alpha3,linestyle='dotted',marker='d',markevery=[-1])
            
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom = 10**-3, top = 20)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2: 
            #VTI
            ax.loglog(xrange23[0:40],CRLBx27[0:40]/CRLBx23[0:40],color='C1', alpha=alpha1,marker='d',markevery=[-1])
            ax.loglog(xrange23[39:80],CRLBx27[39:80]/CRLBx23[39:80],color='C1', alpha=alpha2,marker='d',markevery=[-1])
            ax.loglog(xrange23[79:-1],CRLBx27[79:-1]/CRLBx23[79:-1],color='C1', alpha=alpha3,marker='d',markevery=[-1])
                          
            ax.loglog(xrange3[0:40],CRLBx1[0:40]/CRLBx3[0:40],color='C2', alpha=alpha1,marker='d',markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBx1[39:80]/CRLBx3[39:80],color='C2', alpha=alpha2,marker='d',markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBx1[79:-1]/CRLBx3[79:-1],color='C2', alpha=alpha3,marker='d',markevery=[-1])
            
            ax.loglog(xrange24[0:40],CRLBx28[0:40]/CRLBx24[0:40],color='C3', alpha=alpha1,marker='d',markevery=[-1])
            ax.loglog(xrange24[39:80],CRLBx28[39:80]/CRLBx24[39:80],color='C3', alpha=alpha2,marker='d',markevery=[-1])
            ax.loglog(xrange24[79:-1],CRLBx28[79:-1]/CRLBx24[79:-1],color='C3', alpha=alpha3,marker='d',markevery=[-1])
            
            #VTI Theoretical
            ax.loglog(xrange22[0:40],CRLBxt26[0:40]/CRLBxt22[0:40],color='C4', alpha=alpha1,linestyle='dashed',marker='d',markevery=[-1])
            ax.loglog(xrange22[39:80],CRLBxt26[39:80]/CRLBxt22[39:80],color='C4', alpha=alpha2,linestyle='dashed',marker='d',markevery=[-1])
            ax.loglog(xrange22[79:-1],CRLBxt26[79:-1]/CRLBxt22[79:-1],color='C4', alpha=alpha3,linestyle='dashed',marker='d',markevery=[-1])
            
            #MINFLUX
            ax.loglog(xrange22[0:40],CRLBxM26[0:40]/CRLBxM22[0:40],color='C4', alpha=alpha1,linestyle='dotted',marker='d',markevery=[-1])
            ax.loglog(xrange22[39:80],CRLBxM26[39:80]/CRLBxM22[39:80],color='C4', alpha=alpha2,linestyle='dotted',marker='d',markevery=[-1])
            ax.loglog(xrange22[79:-1],CRLBxM26[79:-1]/CRLBxM22[79:-1],color='C4', alpha=alpha3,linestyle='dotted',marker='d',markevery=[-1])

            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom=7*10**-1, top = 10**3)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'(3 iterations, $m=0.95$)'))
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'(3 iterations, $m=1$)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'(3 iterations, $m=1$)'), linestyle='dotted')
            C4_line = mlines.Line2D([0],[0],color='C4', label=r'$\theta_b$=0 photons/pixel', marker='d')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$\theta_b$=1 photon/pixel', marker='d')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$\theta_b$=4 photons/pixel', marker='d')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$\theta_b$=8 photons/pixel', marker='d')
            C3_line = mlines.Line2D([0],[0],color='C3', label=r'$\theta_b$=12 photons/pixel', marker='d')
     
            ax.legend(handles=[black_cont, black_VTI, black_M , C4_line, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)      

    #################################################################################
    
    elif plotnumber == "S8a":
        # MC-VTI 3 it., alpha = 2
        setnumbers_2 = [207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221]
        xrange_2, CRLBx_2, RMSE_2, RMSEstdev_2 = readdata_thetaI(setnumbers_2, subsets, iters=3)
        
        # MC-VTI 3 it., alpha = 3
        setnumbers_3 = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
        xrange_3, CRLBx_3, RMSE_3, RMSEstdev_3 = readdata_thetaI(setnumbers_3, subsets, iters=3)

        # MC-VTI 3 it., alpha = 4 
        setnumbers_4 = [222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236]
        xrange_4, CRLBx_4, RMSE_4, RMSEstdev_4 = readdata_thetaI(setnumbers_4, subsets, iters=3)
        
        # MC-VTI 1 it.
        params_i, CRLBx_i, RMSE_i, RMSEstdev_i = readdata(79, subsets)
        xrange_i = np.linspace(params_i[2]/(stepsmax), params_i[2], stepsmax)
        CRLBx_i = np.interp(xrange_3, xrange_i, CRLBx_i)
        
        #VTI Theoretical & MINFLUX 3 it., 0 ph./px.
        setnumberst_2 = [239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253]
        CRLBxt_2, CRLBxM_2 = readdataT_thetaI(setnumberst_2)

        setnumberst_3 = [182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]
        CRLBxt_3, CRLBxM_3 = readdataT_thetaI(setnumberst_3)
        
        setnumberst_4 = [254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268]
        CRLBxt_4, CRLBxM_4 = readdataT_thetaI(setnumberst_4)
        
        #VTI Theoretical & MINFLUX 1 it., 0 ph./px.
        CRLBxt_i, CRLBxM_i = readdataT(202)
        CRLBxt_i = np.interp(xrange_3, xrange_i, CRLBxt_i)
        CRLBxM_i = np.interp(xrange_3, xrange_i, CRLBxM_i)
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.loglog(xrange_3, CRLBx_i*10**9, color='C0')
            ax.loglog(xrange_2,CRLBx_2*10**9,color='C1')
            ax.loglog(xrange_3,CRLBx_3*10**9,color='C2')
            ax.loglog(xrange_4,CRLBx_4*10**9,color='C3')
                
            #MAP
            ax.errorbar(xrange_2, RMSE_2*10**9, yerr=RMSEstdev_2*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_3, RMSE_3*10**9, yerr=RMSEstdev_3*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(xrange_4, RMSE_4*10**9, yerr=RMSEstdev_4*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #VTI Theoretical
            ax.loglog(xrange_3,CRLBxt_i*10**9,color='C0', linestyle='dashed')
            ax.loglog(xrange_2,CRLBxt_2*10**9,color='C1', linestyle='dashed')
            ax.loglog(xrange_3,CRLBxt_3*10**9,color='C2', linestyle='dashed')
            ax.loglog(xrange_4,CRLBxt_4*10**9,color='C3', linestyle='dashed')
            
            #MINFLUX
            ax.loglog(xrange_3,CRLBxM_i*10**9,color='C0', linestyle='dotted')
            ax.loglog(xrange_2,CRLBxM_2*10**9,color='C1', linestyle='dotted')
            ax.loglog(xrange_3,CRLBxM_3*10**9,color='C2', linestyle='dotted')
            ax.loglog(xrange_4,CRLBxM_4*10**9,color='C3', linestyle='dotted')
            
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom = 1*10**-4, top = 30)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:  
            #VTI
            ax.loglog(xrange_2, CRLBx_i/CRLBx_2, color='C1')
            ax.loglog(xrange_3, CRLBx_i/CRLBx_3, color='C2')
            ax.loglog(xrange_4, CRLBx_i/CRLBx_4, color='C3')
            
            #MAP
            ax.errorbar(xrange_2, CRLBx_i/RMSE_2, yerr=np.abs(CRLBx_i/RMSE_2 - CRLBx_i/(RMSE_2+RMSEstdev_2)), capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_3, CRLBx_i/RMSE_3, yerr=np.abs(CRLBx_i/RMSE_3 - CRLBx_i/(RMSE_3+RMSEstdev_3)), capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(xrange_4, CRLBx_i/RMSE_4, yerr=np.abs(CRLBx_i/RMSE_4 - CRLBx_i/(RMSE_4+RMSEstdev_4)), capsize=3, color='C3', marker='x', linestyle='None')
            
            
            #VTI Theoretical
            ax.loglog(xrange_2,CRLBxt_i/CRLBxt_2,color='C1', linestyle = 'dashed')
            ax.loglog(xrange_3,CRLBxt_i/CRLBxt_3,color='C2', linestyle = 'dashed')
            ax.loglog(xrange_4,CRLBxt_i/CRLBxt_4,color='C3', linestyle = 'dashed')
       
            #MINFLUX
            ax.loglog(xrange_2,CRLBxM_i/CRLBxM_2,color='C1', linestyle = 'dotted')    
            ax.loglog(xrange_3,CRLBxM_i/CRLBxM_3,color='C2', linestyle = 'dotted')  
            ax.loglog(xrange_4,CRLBxM_i/CRLBxM_4,color='C3', linestyle = 'dotted')  
            
            #Configuration
            ax.set_xlabel(r'Expected amount of signal photons $\theta_I$')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom=4*10**-1, top = 1*10**4)        
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label='RMSE of MAP estimates', marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 2\sigma_{x,k-1}) - \pi$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 4\sigma_{x,k-1}) - \pi$')
     
            ax.legend(handles=[black_cont, black_VTI, black_M , black_dash, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)
          
    #################################################################################
    
    elif plotnumber == "S8b":
        # Opacity settings
        alpha1 = 0.4
        alpha2 = 2/3
        alpha3 = 1

        # Uses datasets 3, 6, 9, 1, 4, 7
        params1, CRLBx1, RMSE1,  RMSEstdev1 = readdata(1, subsets)
        params3, CRLBx3, RMSE3,  RMSEstdev3 = readdata(3, subsets)
        params4, CRLBx4, RMSE4,  RMSEstdev4 = readdata(4, subsets)
        params6, CRLBx6, RMSE6,  RMSEstdev6 = readdata(6, subsets)
        params7, CRLBx7, RMSE7,  RMSEstdev7 = readdata(7, subsets)
        params9, CRLBx9, RMSE9,  RMSEstdev9 = readdata(9, subsets)

        CRLBxt1, CRLBxM1 = readdataT(1)
        CRLBxt3, CRLBxM3 = readdataT(3)
        CRLBxt4, CRLBxM4 = readdataT(4)
        CRLBxt6, CRLBxM6 = readdataT(6)
        CRLBxt7, CRLBxM7 = readdataT(7)
        CRLBxt9, CRLBxM9 = readdataT(9)
        
        xrange1 = np.linspace(params1[2]/(stepsmax), params1[2], stepsmax)
        xrange3 = np.linspace(params3[2]/(stepsmax), params3[2], stepsmax)
        xrange6 = np.linspace(params6[2]/(stepsmax), params6[2], stepsmax)
        xrange9 = np.linspace(params9[2]/(stepsmax), params9[2], stepsmax)
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.loglog(xrange1,CRLBx1*10**9,color='C0', marker='^',markevery=[stepsmax-1])
            
            ax.loglog(xrange6[0:40],CRLBx6[0:40]*10**9,color='C1', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange6[39:80],CRLBx6[39:80]*10**9,color='C1', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange6[79:-1],CRLBx6[79:-1]*10**9,color='C1', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBx3[0:40]*10**9,color='C2', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBx3[39:80]*10**9,color='C2', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBx3[79:-1]*10**9,color='C2', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange9[0:40],CRLBx9[0:40]*10**9,color='C3', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange9[39:80],CRLBx9[39:80]*10**9,color='C3', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange9[79:-1],CRLBx9[79:-1]*10**9,color='C3', alpha=alpha3, marker='d',markevery=[-1])
        
            #VTI Theoretical
            ax.loglog(xrange1,CRLBxt1*10**9,color='C0', linestyle = 'dashed', marker='^', markevery=[stepsmax-1])
            
            ax.loglog(xrange6[0:40],CRLBxt6[0:40]*10**9,color='C1', linestyle = 'dashed', alpha=alpha1, marker='d', markevery=[-1])
            ax.loglog(xrange6[39:80],CRLBxt6[39:80]*10**9,color='C1', linestyle = 'dashed', alpha=alpha2, marker='d', markevery=[-1])
            ax.loglog(xrange6[79:-1],CRLBxt6[79:-1]*10**9,color='C1', linestyle = 'dashed', alpha=alpha3, marker='d', markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBxt3[0:40]*10**9,color='C2', linestyle = 'dashed', alpha=alpha1, marker='d', markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBxt3[39:80]*10**9,color='C2', linestyle = 'dashed', alpha=alpha2, marker='d', markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBxt3[79:-1]*10**9,color='C2', linestyle = 'dashed', alpha=alpha3, marker='d', markevery=[-1])
            
            ax.loglog(xrange9[0:40],CRLBxt9[0:40]*10**9,color='C3', linestyle = 'dashed', alpha=alpha1, marker='d', markevery=[-1])
            ax.loglog(xrange9[39:80],CRLBxt9[39:80]*10**9,color='C3', linestyle = 'dashed', alpha=alpha2, marker='d', markevery=[-1])
            ax.loglog(xrange9[79:-1],CRLBxt9[79:-1]*10**9,color='C3', linestyle = 'dashed', alpha=alpha3, marker='d', markevery=[-1])
            
            #MINFLUX
            ax.loglog(xrange1,CRLBxM1*10**9,color='C0', linestyle = 'dotted', marker='^', markevery=[stepsmax-1])
            
            ax.loglog(xrange6[0:40],CRLBxM6[0:40]*10**9,color='C1', linestyle = 'dotted', alpha=alpha1, marker='d', markevery=[-1])
            ax.loglog(xrange6[39:80],CRLBxM6[39:80]*10**9,color='C1', linestyle = 'dotted', alpha=alpha2, marker='d', markevery=[-1])
            ax.loglog(xrange6[79:-1],CRLBxM6[79:-1]*10**9,color='C1', linestyle = 'dotted', alpha=alpha3, marker='d', markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBxM3[0:40]*10**9,color='C2', linestyle = 'dotted', alpha=alpha1, marker='d', markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBxM3[39:80]*10**9,color='C2', linestyle = 'dotted', alpha=alpha2, marker='d', markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBxM3[79:-1]*10**9,color='C2', linestyle = 'dotted', alpha=alpha3, marker='d', markevery=[-1])
            
            ax.loglog(xrange9[0:40],CRLBxM9[0:40]*10**9,color='C3', linestyle = 'dotted', alpha=alpha1, marker='d', markevery=[-1])
            ax.loglog(xrange9[39:80],CRLBxM9[39:80]*10**9,color='C3', linestyle = 'dotted', alpha=alpha2, marker='d', markevery=[-1])
            ax.loglog(xrange9[79:-1],CRLBxM9[79:-1]*10**9,color='C3', linestyle = 'dotted', alpha=alpha3, marker='d', markevery=[-1])
                   
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom = 7*10**-4, top = 20)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange6[0:40],CRLBx4[0:40]/CRLBx6[0:40], color='C1', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange6[39:80],CRLBx4[39:80]/CRLBx6[39:80], color='C1', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange6[79:-1],CRLBx4[79:-1]/CRLBx6[79:-1], color='C1', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBx1[0:40]/CRLBx3[0:40], color='C2', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBx1[39:80]/CRLBx3[39:80], color='C2', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBx1[79:-1]/CRLBx3[79:-1], color='C2', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange9[0:40],CRLBx7[0:40]/CRLBx9[0:40], color='C3', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange9[39:80],CRLBx7[39:80]/CRLBx9[39:80], color='C3', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange9[79:-1],CRLBx7[79:-1]/CRLBx9[79:-1], color='C3', alpha=alpha3, marker='d',markevery=[-1])
            
            #VTI Theoretical
            ax.loglog(xrange6[0:40],CRLBxt4[0:40]/CRLBxt6[0:40], color='C1', linestyle = 'dashed', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange6[39:80],CRLBxt4[39:80]/CRLBxt6[39:80], color='C1', linestyle = 'dashed', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange6[79:-1],CRLBxt4[79:-1]/CRLBxt6[79:-1], color='C1', linestyle = 'dashed', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBxt1[0:40]/CRLBxt3[0:40], color='C2', linestyle = 'dashed', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBxt1[39:80]/CRLBxt3[39:80], color='C2', linestyle = 'dashed', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBxt1[79:-1]/CRLBxt3[79:-1], color='C2', linestyle = 'dashed', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange9[0:40],CRLBxt7[0:40]/CRLBxt9[0:40], color='C3', linestyle = 'dashed', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange9[39:80],CRLBxt7[39:80]/CRLBxt9[39:80], color='C3', linestyle = 'dashed', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange9[79:-1],CRLBxt7[79:-1]/CRLBxt9[79:-1], color='C3', linestyle = 'dashed', alpha=alpha3, marker='d',markevery=[-1])
            
            #MINFLUX
            ax.loglog(xrange6[0:40],CRLBxM4[0:40]/CRLBxM6[0:40], color='C1', linestyle = 'dotted', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange6[39:80],CRLBxM4[39:80]/CRLBxM6[39:80], color='C1', linestyle = 'dotted', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange6[79:-1],CRLBxM4[79:-1]/CRLBxM6[79:-1], color='C1', linestyle = 'dotted', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange3[0:40],CRLBxM1[0:40]/CRLBxM3[0:40], color='C2', linestyle = 'dotted', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange3[39:80],CRLBxM1[39:80]/CRLBxM3[39:80], color='C2', linestyle = 'dotted', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange3[79:-1],CRLBxM1[79:-1]/CRLBxM3[79:-1], color='C2', linestyle = 'dotted', alpha=alpha3, marker='d',markevery=[-1])
            
            ax.loglog(xrange9[0:40],CRLBxM7[0:40]/CRLBxM9[0:40], color='C3', linestyle = 'dotted', alpha=alpha1, marker='d',markevery=[-1])
            ax.loglog(xrange9[39:80],CRLBxM7[39:80]/CRLBxM9[39:80], color='C3', linestyle = 'dotted', alpha=alpha2, marker='d',markevery=[-1])
            ax.loglog(xrange9[79:-1],CRLBxM7[79:-1]/CRLBxM9[79:-1], color='C3', linestyle = 'dotted', alpha=alpha3, marker='d',markevery=[-1])      
            
            #Configuration
            ax.set_xlabel(r'Cumulative signal photons')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=thetaI/5, right = thetaI+100)
            ax.set_ylim(bottom=7*10**-1, top = 2*10**3)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$', marker='^')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 2\sigma_{x,k-1}) - \pi$', marker='d')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r' $\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$', marker='d')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 4\sigma_{x,k-1}) - \pi$', marker='d')
    
            ax.legend(handles=[black_cont, black_VTI, black_M, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1) 
            

    #################################################################################
    
    elif plotnumber == "S8c":
        params1, CRLBx1, RMSE1,  RMSEstdev1 = readdata(1, subsets)
        params2, CRLBx2, RMSE2,  RMSEstdev2 = readdata(2, subsets)
        params3, CRLBx3, RMSE3,  RMSEstdev3 = readdata(3, subsets)
        params4, CRLBx4, RMSE4,  RMSEstdev4 = readdata(4, subsets)
        params5, CRLBx5, RMSE5,  RMSEstdev5 = readdata(5, subsets)
        params6, CRLBx6, RMSE6,  RMSEstdev6 = readdata(6, subsets)
        params7, CRLBx7, RMSE7,  RMSEstdev7 = readdata(7, subsets)
        params8, CRLBx8, RMSE8,  RMSEstdev8 = readdata(8, subsets)
        params9, CRLBx9, RMSE9,  RMSEstdev9 = readdata(9, subsets)
        
        CRLBxt1, CRLBxM1 = readdataT(1)
        CRLBxt2, CRLBxM2 = readdataT(2)
        CRLBxt3, CRLBxM3 = readdataT(3)
        CRLBxt4, CRLBxM4 = readdataT(4)
        CRLBxt5, CRLBxM5 = readdataT(5)
        CRLBxt6, CRLBxM6 = readdataT(6)
        CRLBxt7, CRLBxM7 = readdataT(7)
        CRLBxt8, CRLBxM8 = readdataT(8)
        CRLBxt9, CRLBxM9 = readdataT(9)
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:      
            #VTI
            ax.semilogy([1,2,3],np.array([CRLBx4[-1]*10**9, CRLBx5[-1]*10**9, CRLBx6[-1]*10**9]),color='C1')
            ax.semilogy([1,2,3],np.array([CRLBx1[-1]*10**9, CRLBx2[-1]*10**9, CRLBx3[-1]*10**9]),color='C2')
            ax.semilogy([1,2,3],np.array([CRLBx7[-1]*10**9, CRLBx8[-1]*10**9, CRLBx9[-1]*10**9]),color='C3')
              
            #MAP
            ax.errorbar([1,2,3], np.array([RMSE4[-1]*10**9, RMSE5[-1]*10**9, RMSE6[-1]*10**9]), yerr=np.array([RMSEstdev4[-1]*10**9, RMSEstdev5[-1]*10**9, RMSEstdev6[-1]*10**9]), capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar([1,2,3], np.array([RMSE1[-1]*10**9, RMSE2[-1]*10**9, RMSE3[-1]*10**9]), yerr=np.array([RMSEstdev1[-1]*10**9, RMSEstdev2[-1]*10**9, RMSEstdev3[-1]*10**9]), capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar([1,2,3], np.array([RMSE7[-1]*10**9, RMSE8[-1]*10**9, RMSE9[-1]*10**9]), yerr=np.array([RMSEstdev7[-1]*10**9, RMSEstdev8[-1]*10**9, RMSEstdev9[-1]*10**9]), capsize=3, color='C3', marker='x', linestyle='None')
    
            #VTI Theoretical
            ax.semilogy([1,2,3],np.array([CRLBxt4[-1]*10**9, CRLBxt5[-1]*10**9, CRLBxt6[-1]*10**9]),color='C1', linestyle='dashed')
            ax.semilogy([1,2,3],np.array([CRLBxt1[-1]*10**9, CRLBxt2[-1]*10**9, CRLBxt3[-1]*10**9]),color='C2', linestyle='dashed')
            ax.semilogy([1,2,3],np.array([CRLBxt7[-1]*10**9, CRLBxt8[-1]*10**9, CRLBxt9[-1]*10**9]),color='C3', linestyle='dashed')
    
            #MINFLUX
            ax.semilogy([1,2,3],np.array([CRLBxM4[-1]*10**9, CRLBxM5[-1]*10**9, CRLBxM6[-1]*10**9]),color='C1', linestyle='dotted')
            ax.semilogy([1,2,3],np.array([CRLBxM1[-1]*10**9, CRLBxM2[-1]*10**9, CRLBxM3[-1]*10**9]),color='C2', linestyle='dotted')
            ax.semilogy([1,2,3],np.array([CRLBxM7[-1]*10**9, CRLBxM8[-1]*10**9, CRLBxM9[-1]*10**9]),color='C3', linestyle='dotted')
            
            #Configuration
            ax.set_xlabel(r'Amount of iterations')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=0.8, right = 3.2)
            ax.set_ylim(bottom = 10**-3, top = 20)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
            ax.set_xticks([1, 2, 3])
            ax.xaxis.set_ticklabels([1, 2, 3])
        
        
        if panel==2:
            #VTI
            ax.semilogy([1,2,3],CRLBx4[-1]/np.array([CRLBx4[-1], CRLBx5[-1], CRLBx6[-1]]), color='C1')
            ax.semilogy([1,2,3],CRLBx1[-1]/np.array([CRLBx1[-1], CRLBx2[-1], CRLBx3[-1]]), color='C2')
            ax.semilogy([1,2,3],CRLBx7[-1]/np.array([CRLBx7[-1], CRLBx8[-1], CRLBx9[-1]]), color='C3')
            
            #MAP
            ax.errorbar([1,2,3], CRLBx4[-1]/np.array([RMSE4[-1], RMSE5[-1], RMSE6[-1]]), yerr=np.abs(CRLBx4[-1]/np.array([RMSE4[-1], RMSE5[-1], RMSE6[-1]]) - CRLBx4[-1]/(np.array([RMSE4[-1], RMSE5[-1], RMSE6[-1]])+np.array([RMSEstdev4[-1], RMSEstdev5[-1], RMSEstdev6[-1]]))), capsize=3, marker='x', color='C1', linestyle='None')
            ax.errorbar([1,2,3], CRLBx1[-1]/np.array([RMSE1[-1], RMSE2[-1], RMSE3[-1]]), yerr=np.abs(CRLBx1[-1]/np.array([RMSE1[-1], RMSE2[-1], RMSE3[-1]]) - CRLBx1[-1]/(np.array([RMSE1[-1], RMSE2[-1], RMSE3[-1]])+np.array([RMSEstdev1[-1], RMSEstdev2[-1], RMSEstdev3[-1]]))), capsize=3, marker='x', color='C2', linestyle='None')
            ax.errorbar([1,2,3], CRLBx7[-1]/np.array([RMSE7[-1], RMSE8[-1], RMSE9[-1]]), yerr=np.abs(CRLBx7[-1]/np.array([RMSE7[-1], RMSE8[-1], RMSE9[-1]]) - CRLBx7[-1]/(np.array([RMSE7[-1], RMSE8[-1], RMSE9[-1]])+np.array([RMSEstdev7[-1], RMSEstdev8[-1], RMSEstdev9[-1]]))), capsize=3, marker='x', color='C3', linestyle='None')
    
            #VTI Theoretical
            ax.semilogy([1,2,3],CRLBxt4[-1]/np.array([CRLBxt4[-1], CRLBxt5[-1], CRLBxt6[-1]]), color='C1', linestyle='dashed')
            ax.semilogy([1,2,3],CRLBxt1[-1]/np.array([CRLBxt1[-1], CRLBxt2[-1], CRLBxt3[-1]]), color='C2', linestyle='dashed')
            ax.semilogy([1,2,3],CRLBxt7[-1]/np.array([CRLBxt7[-1], CRLBxt8[-1], CRLBxt9[-1]]), color='C3', linestyle='dashed')        
    
            #MINFLUX
            ax.semilogy([1,2,3],CRLBxM4[-1]/np.array([CRLBxM4[-1], CRLBxM5[-1], CRLBxM6[-1]]), color='C1', linestyle='dotted')
            ax.semilogy([1,2,3],CRLBxM1[-1]/np.array([CRLBxM1[-1], CRLBxM2[-1], CRLBxM3[-1]]), color='C2', linestyle='dotted')
            ax.semilogy([1,2,3],CRLBxM7[-1]/np.array([CRLBxM7[-1], CRLBxM8[-1], CRLBxM9[-1]]), color='C3', linestyle='dotted') 
    
            #Configuration
            ax.set_xlabel(r'Amount of iterations')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1\ \mathrm{iter.}}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=0.8, right = 3.2)
            ax.set_ylim(bottom=7*10**-1, top = 10**3)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
            ax.set_xticks([1, 2, 3])
            ax.xaxis.set_ticklabels([1, 2, 3])
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_I = 2000$ ph.,' + '\n' + r'$\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label='RMSE of MAP estimates', marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_I = 2000$ ph.,' + '\n' + r'$\theta_b = 0$ ph./px.)'), linestyle='dashed')
            black_M = mlines.Line2D([0], [0],color='black', label=('CRLB approximation (12)'+ '\n' +r'($m=1$, $\theta_I = 2000$ ph.,' + '\n' + r'$\theta_b = 0$ ph./px.)'), linestyle='dotted')
            C0_line = mlines.Line2D([0], [0],color='C1', label=r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 2\sigma_{x,k-1}) - \pi$')
            C1_line = mlines.Line2D([0], [0],color='C2', label=r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
            C2_line = mlines.Line2D([0], [0],color='C3', label=r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 4\sigma_{x,k-1}) - \pi$')
    
            ax.legend(handles=[black_cont, black_VTI, black_M, black_dash, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

    #################################################################################
    
    if plotnumber == "S10":
        roisize,m,kx,ky,sigmap,other = vti.imgparams()
        omega = other[2]
        
        # MC-VTI 2 it., m = 0.8
        setnumbers_8 = [403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417]
        xrange_8, CRLBx_8, RMSE_8, RMSEstdev_8, alpharange_8, CRLBx_8_1i = readdata_thetaI(setnumbers_8, subsets, iters=2, readalpha=True)
        phirange_8 = 2*omega*alpharange_8*CRLBx_8_1i

        # MC-VTI 2 it., m = 0.9
        setnumbers_9 = [418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432]
        xrange_9, CRLBx_9, RMSE_9, RMSEstdev_9, alpharange_9, CRLBx_9_1i = readdata_thetaI(setnumbers_9, subsets, iters=2, readalpha=True)
        phirange_9 = 2*omega*alpharange_9*CRLBx_9_1i
        
        # MC-VTI 2 it., m = 0.95
        setnumbers_95 = [319, 324, 329, 334, 339, 344, 349, 350, 351, 352, 353, 354, 355, 356, 357]
        xrange_95, CRLBx_95, RMSE_95, RMSEstdev_95, alpharange_95, CRLBx_95_1i = readdata_thetaI(setnumbers_95, subsets, iters=2, readalpha=True)
        phirange_95 = 2*omega*alpharange_95*CRLBx_95_1i
        
        # MC-VTI 2 it., m = 1
        setnumbers_10 = [433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447]
        xrange_10, CRLBx_10, RMSE_10, RMSEstdev_10, alpharange_10, CRLBx_10_1i = readdata_thetaI(setnumbers_10, subsets, iters=2, readalpha=True)
        phirange_10 = 2*omega*alpharange_10*CRLBx_10_1i
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(10, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.semilogy(alpharange_8,CRLBx_8*10**9,color='C0')
            ax.semilogy(alpharange_9,CRLBx_9*10**9,color='C1')
            ax.semilogy(alpharange_95,CRLBx_95*10**9,color='C2')
            ax.semilogy(alpharange_10,CRLBx_10*10**9,color='C3')
                
            #MAP
            ax.errorbar(alpharange_8, RMSE_8*10**9, yerr=RMSEstdev_8*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(alpharange_9, RMSE_9*10**9, yerr=RMSEstdev_9*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(alpharange_95, RMSE_95*10**9, yerr=RMSEstdev_95*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(alpharange_10[1:-1], RMSE_10[1:-1]*10**9, yerr=RMSEstdev_10[1:-1]*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #Configuration
            ax.set_xlabel(r'Aggressiveness parameter $\alpha$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_ylim(bottom=10**-1)  
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.semilogy(phirange_8,CRLBx_8*10**9,color='C0')
            ax.semilogy(phirange_9,CRLBx_9*10**9,color='C1')
            ax.semilogy(phirange_95,CRLBx_95*10**9,color='C2')
            ax.semilogy(phirange_10,CRLBx_10*10**9,color='C3')
            
            #MAP
            ax.errorbar(phirange_8, RMSE_8*10**9, yerr=RMSEstdev_8*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(phirange_9, RMSE_9*10**9, yerr=RMSEstdev_9*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(phirange_95, RMSE_95*10**9, yerr=RMSEstdev_95*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(phirange_10[1:-1], RMSE_10[1:-1]*10**9, yerr=RMSEstdev_10[1:-1]*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #Configuration
            ax.set_xlabel(r'x-phase between pattern minima $\phi_{x,2}^+-\phi_{x,2}^-$ [rad]')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()  
            ax.set_ylim(bottom=10**-1)  
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'(2 iterations, $\theta_b$=8 photons/pixel)'))
            black_dash = mlines.Line2D([0], [0],color='black', label='RMSE of MAP estimates', marker='x', linestyle='None')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$m=0.8$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$m=0.9$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$m=0.95$')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'$m=1.0$')
     
            ax.legend(handles=[black_cont, black_dash, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)        

    #################################################################################
    
    if plotnumber == "S11":
        roisize,m,kx,ky,sigmap,other = vti.imgparams()
        omega = other[2]
        
        # MC-VTI 2 it., 1 ph./px.
        setnumbers_1 = [358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372]
        xrange_1, CRLBx_1, RMSE_1, RMSEstdev_1, alpharange_1, CRLBx_1_1i = readdata_thetaI(setnumbers_1, subsets, iters=2, readalpha=True)
        phirange_1 = 2*omega*alpharange_1*CRLBx_1_1i

        # MC-VTI 2 it., 4 ph./px
        setnumbers_4 = [373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387]
        xrange_4, CRLBx_4, RMSE_4, RMSEstdev_4, alpharange_4, CRLBx_4_1i = readdata_thetaI(setnumbers_4, subsets, iters=2, readalpha=True)
        phirange_4 = 2*omega*alpharange_4*CRLBx_4_1i
        
        # MC-VTI 2 it., 8 ph./px. 
        setnumbers_8 = [319, 324, 329, 334, 339, 344, 349, 350, 351, 352, 353, 354, 355, 356, 357]
        xrange_8, CRLBx_8, RMSE_8, RMSEstdev_8, alpharange_8, CRLBx_8_1i = readdata_thetaI(setnumbers_8, subsets, iters=2, readalpha=True)
        phirange_8 = 2*omega*alpharange_8*CRLBx_8_1i

        # MC-VTI 3 it., 12 ph./px. 
        setnumbers_12 = [388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402]
        xrange_12, CRLBx_12, RMSE_12, RMSEstdev_12, alpharange_12, CRLBx_12_1i = readdata_thetaI(setnumbers_12, subsets, iters=2, readalpha=True)        
        phirange_12 = 2*omega*alpharange_12*CRLBx_12_1i
        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(10, 5, forward=True)
        
        if panel==1:
            #VTI
            ax.semilogy(alpharange_1,CRLBx_1*10**9,color='C0')
            ax.semilogy(alpharange_4,CRLBx_4*10**9,color='C1')
            ax.semilogy(alpharange_8,CRLBx_8*10**9,color='C2')
            ax.semilogy(alpharange_12,CRLBx_12*10**9,color='C3')
                
            #MAP
            ax.errorbar(alpharange_1, RMSE_1*10**9, yerr=RMSEstdev_1*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(alpharange_4, RMSE_4*10**9, yerr=RMSEstdev_4*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(alpharange_8, RMSE_8*10**9, yerr=RMSEstdev_8*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(alpharange_12, RMSE_12*10**9, yerr=RMSEstdev_12*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #Configuration
            ax.set_xlabel(r'Aggressiveness parameter $\alpha$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.semilogy(phirange_1,CRLBx_1*10**9,color='C0')
            ax.semilogy(phirange_4,CRLBx_4*10**9,color='C1')
            ax.semilogy(phirange_8,CRLBx_8*10**9,color='C2')
            ax.semilogy(phirange_12,CRLBx_12*10**9,color='C3')
            
            #MAP
            ax.errorbar(phirange_1, RMSE_1*10**9, yerr=RMSEstdev_1*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(phirange_4, RMSE_4*10**9, yerr=RMSEstdev_4*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(phirange_8, RMSE_8*10**9, yerr=RMSEstdev_8*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(phirange_12, RMSE_12*10**9, yerr=RMSEstdev_12*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #Configuration
            ax.set_xlabel(r'x-phase between pattern minima $\phi_{x,2}^+-\phi_{x,2}^-$ [rad]')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()       
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'(2 iterations, $m=0.95$)'))
            black_dash = mlines.Line2D([0], [0],color='black', label='RMSE of MAP estimates', marker='x', linestyle='None')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'$\theta_b$=1 photon/pixel')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'$\theta_b$=4 photons/pixel')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'$\theta_b$=8 photons/pixel')
            C3_line = mlines.Line2D([0],[0],color='C3', label=r'$\theta_b$=12 photons/pixel')
     
            ax.legend(handles=[black_cont, black_dash, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)


#Figure 2d
plot("2d",1)
plot("2d",2)
plot("2d",0)

#Figure 2e
plot("2e",1)
plot("2e",2)
plot("2e",0)

#Figure S2a
plot("S2a",1)
plot("S2a",2)
plot("S2a",0)

#Figure S2b
plot("S2b",1)
plot("S2b",2)
plot("S2b",0)

#Figure S5
plot("S5",1)
plot("S5",2)
plot("S5",0)

#Figure S6a
plot("S6a",1)
plot("S6a",2)
plot("S6a",0)

#Figure S6b
plot("S6b",1)
plot("S6b",2)
plot("S6b",0)

#Figure S6c
plot("S6c",1)
plot("S6c",2)
plot("S6c",0)

#Figure S7a
plot("S7a",1)
plot("S7a",2)
plot("S7a",0)

#Figure S7b
plot("S7b",1)
plot("S7b",2)
plot("S7b",0)

#Figure S8a
plot("S8a",1)
plot("S8a",2)
plot("S8a",0)

#Figure S8b
plot("S8b",1)
plot("S8b",2)
plot("S8b",0)

#Figure S8c
plot("S8c",1)
plot("S8c",2)
plot("S8c",0)

#Figure S10a
plot("S10",1)
plot("S10",0)

#Figure S10b
plot("S10",2)

#Figure S11a
plot("S11",1)
plot("S11",0)

#Figure S11b
plot("S11",2)