# -*- coding: utf-8 -*-
"""
Reading data, processing and plotting of Figures
    3d, 3e, 
    S9a, S9b, S9c, S9d
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import VTI_helper as vti
import matplotlib as mpl

new_rc_params = {
"font.size": 12}

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

def readdata(setnumber, subsets, stepsmax):
    '''
    Reads VTI and MAP data from ./SimData/
    
    Input:
        setnumber: number of dataset as given in Datasets.xlsx
        subsets: amount of subsets used for RMSE computation
        stepsmax: amount of intermediate steps used in the datasets
        
    Output:
        params: Imaging parameters for dataset
        CRLBx: CRLB/VTI for dataset
        thetaMAP: stack of MAP estimates for dataset
        RMSE: RMSE of thetaMAP
        RMSE_stdev: stdev of RMSE of thetaMAP
        illum: fraction of the photon budget used in each iteration
        stdillum: standard deviation of the photon budget used in each iteration
        xrange: realized photon count used in each iteration
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
    
    saveCRLB = 'FT-VTIx-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    savethetaMAP = 'FT-thetaMAP-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m))+ '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    saveillum = 'FT-mod-iter-' + str(itermax) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    savetheta = 'FT-theta-iter-' + str(itermax) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(thetab))+ '.npy'
    
    thetaMAP = np.load(loadpath+savethetaMAP)
    CRLBx = np.load(loadpath+saveCRLB)
    illum = np.load(loadpath+saveillum)
    theta_per_spot=np.load(loadpath+savetheta)

    #Cumulate illumination
    for i in range(len(illum[0])):
        if i > 0:
            illum[:,i] += illum[:,i-1]
    
    stdillum = np.std(illum,axis=0)
    
    xrange = np.linspace(params[2]*np.average(illum[:,0])/(stepsmax/itermax), params[2]*np.average(illum[:,0]), int(stepsmax/itermax))
    
    for i in range(len(illum[0])):
        if i > 0:
            xrange = np.append(xrange, params[2]*np.average(illum[:,i-1]) + np.linspace(params[2]*np.average(illum[:,i]-illum[:,i-1])/(stepsmax/itermax), params[2]*np.average(illum[:,i]-illum[:,i-1]), int(stepsmax/itermax)))

    RMSE_mat = np.zeros((subsets,itermax));

    for iteration in range(itermax):
        for subset in range(subsets):
            RMSE_mat[subset,iteration]=RMSE(thetaMAP[subset*len(thetaMAP)//subsets:(subset+1)*len(thetaMAP)//subsets,iteration,0],theta_per_spot[subset*len(thetaMAP)//subsets:(subset+1)*len(thetaMAP)//subsets,0])*delta_x 
    
    RMSE_val = np.average(RMSE_mat,axis=0)
    RMSE_stdev = np.std(RMSE_mat,axis=0)
    
    return params,CRLBx,RMSE_val,RMSE_stdev,illum,stdillum, xrange

def readdataT(setnumber,stepsmax):
    '''
    Reads data of analytical approximation of the VTI from ./SimData/
    
    Input:
        setnumber: number of dataset as given in Datasets.xlsx
        stepsmax: amount of intermediate steps used in the datasets
        
    Output:
        CRLBxt: Analytical approximation of the VTI for dataset
        illum: fraction of the photon budget used in each iteration
        xrange: realized photon count used in each iteration
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
    params = [itermax,alpha,thetaI,m,thetab]
    
    saveCRLBt = 'FT-VTIxT-iter-' + str(int(itermax)) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(0))+ '.npy'
    CRLBxt = np.load(loadpath+saveCRLBt)
    saveillum = 'FT-modT-iter-' + str(itermax) + '-thetaI-' + str(int(thetaI)) + '-m-' + str(int(100)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(0))+ '.npy'
    illum = np.load(loadpath+saveillum)

    #Cumulate illumination
    for i in range(len(illum)):
        if i > 0:
            illum[i] += illum[i-1]
    
    xrange = np.linspace(params[2]*np.average(illum[0])/(stepsmax/itermax), params[2]*np.average(illum[0]), int(stepsmax/itermax))
    
    for i in range(len(illum)):
        if i > 0:
            xrange = np.append(xrange, params[2]*np.average(illum[i-1]) + np.linspace(params[2]*np.average(illum[i]-illum[i-1])/(stepsmax/itermax), params[2]*np.average(illum[i]-illum[i-1]), int(stepsmax/itermax)))    
    
    return CRLBxt,illum,xrange

def readdata_thetaI(setnumbers, subsets, iters, stepsmax=120):
    '''
    Reads VTI and MAP data from ./SimData/, for a range of datasets with different thetaI values.
    
    Input:
        setnumbers: list of dataset numbers as given in Datasets.xlsx
        subsets: amount of subsets used for RMSE computation
        iters: Amount of iterations corresponding to the dataset.
        
    Optional input:
        stepsmax: amount of intermediate steps used in the datasets
        
    Output:
        xrange: Range of thetaI values
        xrange_adj: Range of realized photon count values
        CRLBx: CRLB/VTI for datasets
        RMSE: RMSE of thetaMAP
        RMSE_stdev: stdev of RMSE of thetaMAP        
    '''
    params_mat = np.empty((len(setnumbers),5))
    CRLBx_mat = np.empty((len(setnumbers),stepsmax))
    RMSE_mat = np.empty((len(setnumbers),iters))
    RMSEstdev_mat = np.empty((len(setnumbers),iters))
    xrange_mat = np.empty((len(setnumbers),stepsmax))
    
    for i in range(len(setnumbers)):
        params, CRLBx, RMSE,  RMSEstdev, illum, stdillum, xrange = readdata(setnumbers[i], subsets, stepsmax=stepsmax)
        params_mat[i,:] = params
        CRLBx_mat[i,:] = CRLBx
        RMSE_mat[i,:] = RMSE
        RMSEstdev_mat[i,:] = RMSEstdev
        xrange_mat[i,:] = xrange
        
    xrange = params_mat[:,2]
    xrange_adj = xrange_mat[:,-1]
    CRLBx = CRLBx_mat[:,-1]
    RMSE = RMSE_mat[:,-1]
    RMSEstdev = RMSEstdev_mat[:,-1]
    
    return xrange, xrange_adj, CRLBx, RMSE, RMSEstdev

def readdataT_thetaI(setnumbers, stepsmax=120):
    '''
    Reads data of analytical approximation of the VTI from ./SimData/, 
    for a range of datasets with different thetaI values.
    
    Input:
        setnumbers: list of dataset numbers as given in Datasets.xlsx
        
    Optional input:
        stepsmax: amount of intermediate steps used in the datasets
           
    Output:
        CRLBxt: Analytical approximation of the VTI for datasets
        xrange: Range of thetaI values
    '''
    CRLBxt_mat = np.empty((len(setnumbers),stepsmax))
    xrange_mat = np.empty((len(setnumbers),stepsmax))
    
    for i in range(len(setnumbers)):
        CRLBxt, _, xrange = readdataT(setnumbers[i],stepsmax)
        CRLBxt_mat[i,:] = CRLBxt
        xrange_mat[i,:] = xrange
        
    CRLBxt = CRLBxt_mat[:,-1]
    xrange = xrange_mat[:,-1]
    return CRLBxt, xrange

def plot(plotnumber, panel):
    '''
    Reading data, processing and plotting of Figures 3d, 3e, S9a, S9b, S9c, S9d
    
    Input:
        plotnumber: Figure number from the article, input as a string
        panel: Panel number. Use 1 for the left panel, 2 for the right panel and 0 for the legend.
    '''
    
    subsets = 200
    stepsmax = 120
        
    #################################################################################
    
    if plotnumber == "3d":        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        # 1 iterations
        setnumbers_1i = [37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
        xrange_1i, _, CRLBx_1i, RMSE_1i, RMSEstdev_1i = readdata_thetaI(setnumbers_1i, subsets, iters=1)

        # 2 iterations
        setnumbers_2i = [38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80]
        xrange_2i, _, CRLBx_2i, RMSE_2i, RMSEstdev_2i = readdata_thetaI(setnumbers_2i, subsets, iters=2)
        
        # 3 iterations
        setnumbers_3i = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
        xrange_3i, _, CRLBx_3i, RMSE_3i, RMSEstdev_3i = readdata_thetaI(setnumbers_3i, subsets, iters=3)     

        #VTI Theoretical
        CRLBxt_1i, _ = readdataT_thetaI(setnumbers_1i)
        CRLBxt_2i, _ = readdataT_thetaI(setnumbers_2i)
        CRLBxt_3i, _ = readdataT_thetaI(setnumbers_3i)      

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
            
            #Configuration
            ax.set_xlabel(r'Expected signal photon budget $\theta_I$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom = 3*10**-1, top = 30)
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
            
            #Configuration
            ax.set_xlabel(r'Expected signal photon budget $\theta_I$')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1 iter.}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom=3*10**-1, top = 4)    
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label=('RMSE of MAP estimates' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'), marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'2 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
    
            ax.legend(handles=[black_cont, black_VTI, black_dash, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

    #################################################################################
    
    if plotnumber == "3e":        
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        # 1 iterations
        setnumbers_1i = [37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
        xrange_1i, _, CRLBx_1i, RMSE_1i, RMSEstdev_1i = readdata_thetaI(setnumbers_1i, subsets, iters=1)

        # 2 iterations, alpha = 2
        setnumbers_2i = [207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221]
        xrange_2i, _, CRLBx_2i, RMSE_2i, RMSEstdev_2i = readdata_thetaI(setnumbers_2i, subsets, iters=3)
        
        # 3 iterations, alpha = 3
        setnumbers_3i = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
        xrange_3i, _, CRLBx_3i, RMSE_3i, RMSEstdev_3i = readdata_thetaI(setnumbers_3i, subsets, iters=3)    
        
        # 3 iterations, alpha = 4
        setnumbers_4i = [222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236]
        xrange_4i, _, CRLBx_4i, RMSE_4i, RMSEstdev_4i = readdata_thetaI(setnumbers_4i, subsets, iters=3)  

        #VTI Theoretical
        CRLBxt_1i, xranget_1i = readdataT_thetaI(setnumbers_1i)
        CRLBxt_2i, xranget_2i = readdataT_thetaI(setnumbers_2i)
        CRLBxt_3i, xranget_3i = readdataT_thetaI(setnumbers_3i) 
        CRLBxt_4i, xranget_4i = readdataT_thetaI(setnumbers_4i)     

        if panel==1:
        #VTI
            ax.loglog(xrange_1i,CRLBx_1i*10**9,color='C0')
            ax.loglog(xrange_2i,CRLBx_2i*10**9,color='C1')
            ax.loglog(xrange_3i,CRLBx_3i*10**9,color='C2')
            ax.loglog(xrange_4i,CRLBx_4i*10**9,color='C3')
    
            #MAP
            ax.errorbar(xrange_1i, RMSE_1i*10**9, yerr=RMSEstdev_1i*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax.errorbar(xrange_2i, RMSE_2i*10**9, yerr=RMSEstdev_2i*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_3i, RMSE_3i*10**9, yerr=RMSEstdev_3i*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(xrange_4i, RMSE_4i*10**9, yerr=RMSEstdev_4i*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #VTI Theoretical
            ax.loglog(xrange_1i,CRLBxt_1i*10**9,color='C0', linestyle = 'dashed')
            ax.loglog(xrange_2i,CRLBxt_2i*10**9,color='C1', linestyle = 'dashed')
            ax.loglog(xrange_3i,CRLBxt_3i*10**9,color='C2', linestyle = 'dashed')
            ax.loglog(xrange_4i,CRLBxt_4i*10**9,color='C3', linestyle = 'dashed')
            
            #Configuration
            ax.set_xlabel(r'Expected signal photon budget $\theta_I$')
            ax.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom = 3*10**-1, top = 30)
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==2:
            #VTI
            ax.loglog(xrange_2i,CRLBx_1i/CRLBx_2i,color='C1')
            ax.loglog(xrange_3i,CRLBx_1i/CRLBx_3i,color='C2')
            ax.loglog(xrange_4i,CRLBx_1i/CRLBx_4i,color='C3')
    
            #MAP
            ax.errorbar(xrange_2i, CRLBx_1i/RMSE_2i, yerr=np.abs(CRLBx_1i/RMSE_2i - CRLBx_1i/(RMSE_2i+RMSEstdev_2i)), capsize=3, color='C1', marker='x', linestyle='None')
            ax.errorbar(xrange_3i, CRLBx_1i/RMSE_3i, yerr=np.abs(CRLBx_1i/RMSE_3i - CRLBx_1i/(RMSE_3i+RMSEstdev_3i)), capsize=3, color='C2', marker='x', linestyle='None')
            ax.errorbar(xrange_4i, CRLBx_1i/RMSE_4i, yerr=np.abs(CRLBx_1i/RMSE_4i - CRLBx_1i/(RMSE_4i+RMSEstdev_4i)), capsize=3, color='C3', marker='x', linestyle='None')
    
            #VTI Theoretical
            ax.loglog(xrange_2i,CRLBxt_1i/CRLBxt_2i,color='C1', linestyle = 'dashed')
            ax.loglog(xrange_3i,CRLBxt_1i/CRLBxt_3i,color='C2', linestyle = 'dashed')
            ax.loglog(xrange_4i,CRLBxt_1i/CRLBxt_4i,color='C3', linestyle = 'dashed')
            
            #Configuration
            ax.set_xlabel(r'Expected signal photon budget $\theta_I$')
            ax.set_ylabel(r'Improvement ratio $\sigma_{x, 1 iter.}/\sigma_{x}$')
            ax.grid()
            ax.set_xlim(left=200, right = 10000)
            ax.set_ylim(bottom=3*10**-1, top = 4)    
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label=('RMSE of MAP estimates' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'), marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 2\sigma_{x,k-1}) - \pi$')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 4\sigma_{x,k-1}) - \pi$')
    
            ax.legend(handles=[black_cont, black_VTI, black_dash, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)

        
    #################################################################################
    
    if plotnumber == "S9a":        
        fig, (ax1) = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            # 1 iterations
            setnumbers_1i = [37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
            _, xrange_1i, CRLBx_1i, RMSE_1i, RMSEstdev_1i = readdata_thetaI(setnumbers_1i, subsets, iters=1)
    
            # 2 iterations
            setnumbers_2i = [38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80]
            _, xrange_2i, CRLBx_2i, RMSE_2i, RMSEstdev_2i = readdata_thetaI(setnumbers_2i, subsets, iters=2)
            
            # 3 iterations
            setnumbers_3i = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
            _, xrange_3i, CRLBx_3i, RMSE_3i, RMSEstdev_3i = readdata_thetaI(setnumbers_3i, subsets, iters=3)     
    
            #VTI Theoretical
            CRLBxt_1i, xranget_1i = readdataT_thetaI(setnumbers_1i)
            CRLBxt_2i, xranget_2i = readdataT_thetaI(setnumbers_2i)
            CRLBxt_3i, xranget_3i = readdataT_thetaI(setnumbers_3i)       
    
            #VTI
            ax1.loglog(xrange_1i,CRLBx_1i*10**9,color='C0')
            ax1.loglog(xrange_2i,CRLBx_2i*10**9,color='C1')
            ax1.loglog(xrange_3i,CRLBx_3i*10**9,color='C2')
    
            #MAP
            ax1.errorbar(xrange_1i, RMSE_1i*10**9, yerr=RMSEstdev_1i*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax1.errorbar(xrange_2i, RMSE_2i*10**9, yerr=RMSEstdev_2i*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax1.errorbar(xrange_3i, RMSE_3i*10**9, yerr=RMSEstdev_3i*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            
            #VTI Theoretical
            ax1.loglog(xranget_1i,CRLBxt_1i*10**9,color='C0', linestyle = 'dashed')
            ax1.loglog(xranget_2i,CRLBxt_2i*10**9,color='C1', linestyle = 'dashed')
            ax1.loglog(xranget_3i,CRLBxt_3i*10**9,color='C2', linestyle = 'dashed')
            
            #Configuration
            ax1.set_xlabel(r'Expected amount of signal photons'+'\nunder illumination Eq. 4')
            ax1.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax1.grid()
            ax1.set_xlim(left=200, right = 10000)
            ax1.set_ylim(bottom = 3*10**-1, top = 30)
            ax1.tick_params(axis='both', which='major')
            ax1.tick_params(axis='both', which='minor')
        
    #################################################################################
    
    if plotnumber == "S9b":
        thetaI=2000
        
        #Opacity scale settings
        alpha1 = 0.4
        alpha2 = 2/3
        alpha3 = 1
        
        fig, (ax1) = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            params1, CRLBx1, RMSE1,  RMSEstdev1, illum1, stdillum1, xrange1  = readdata(1, subsets, stepsmax)
            params2, CRLBx2, RMSE2,  RMSEstdev2, illum2, stdillum2, xrange2  = readdata(2, subsets, stepsmax)
            params3, CRLBx3, RMSE3,  RMSEstdev3, illum3, stdillum3, xrange3  = readdata(3, subsets, stepsmax)        
            
            CRLBxt1,illumt1,xranget1=readdataT(1,stepsmax)
            CRLBxt2,illumt2,xranget2=readdataT(2,stepsmax)
            CRLBxt3,illumt3,xranget3=readdataT(3,stepsmax)
            
            #VTI
            ax1.loglog(xrange1, CRLBx1*10**9,color='C0', alpha = alpha3, marker = '^', markevery=[-1])
            
            ax1.loglog(xrange2[0:60], CRLBx2[0:60]*10**9,color='C1', alpha = alpha2, marker = 'o', markevery=[-1])
            ax1.loglog(xrange2[59:-1], CRLBx2[59:-1]*10**9,color='C1', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax1.loglog(xrange3[0:40], CRLBx3[0:40]*10**9,color='C2', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xrange3[39:80], CRLBx3[39:80]*10**9,color='C2', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xrange3[79:-1], CRLBx3[79:-1]*10**9,color='C2', alpha = alpha3, marker = 'd', markevery=[-1])
            
            #VTI Theoretical
            ax1.loglog(xranget1, CRLBxt1*10**9,color='C0', linestyle = 'dashed', alpha = alpha3, marker = '^', markevery=[-1])
            
            ax1.loglog(xranget2[0:60], CRLBxt2[0:60]*10**9,color='C1', linestyle='dashed', alpha = alpha2, marker = 'o', markevery=[-1])
            ax1.loglog(xranget2[59:-1], CRLBxt2[59:-1]*10**9,color='C1', linestyle='dashed', alpha = alpha3, marker = 'o', markevery=[-1])
            
            ax1.loglog(xranget3[0:40], CRLBxt3[0:40]*10**9,color='C2', linestyle='dashed', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xranget3[39:80], CRLBxt3[39:80]*10**9,color='C2', linestyle='dashed', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xranget3[79:-1], CRLBxt3[79:-1]*10**9,color='C2', linestyle='dashed', alpha = alpha3, marker = 'd', markevery=[-1])
            
            #Configuration
            ax1.set_xlabel(r'Cumulative signal photons')
            ax1.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax1.grid()
            ax1.set_xlim(left=thetaI/5, right = thetaI+100)
            ax1.set_ylim(bottom = 3*10**-1, top = 30)
            ax1.tick_params(axis='both', which='major')
            ax1.tick_params(axis='both', which='minor')
           
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label=('RMSE of MAP estimates'), marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$', marker = '^')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'2 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$', marker = 'o')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$', marker = 'd')
    
            ax1.legend(handles=[black_cont, black_VTI, black_dash, C0_line, C1_line, C2_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)
        
    #################################################################################
    
    if plotnumber == "S9c":        
        fig, (ax1) = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        if panel==1:
            # 1 iterations
            setnumbers_1i = [37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
            _, xrange_1i, CRLBx_1i, RMSE_1i, RMSEstdev_1i = readdata_thetaI(setnumbers_1i, subsets, iters=1)
    
            # 2 iterations, alpha = 2
            setnumbers_2i = [207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221]
            _, xrange_2i, CRLBx_2i, RMSE_2i, RMSEstdev_2i = readdata_thetaI(setnumbers_2i, subsets, iters=3)
            
            # 3 iterations, alpha = 3
            setnumbers_3i = [39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81]
            _, xrange_3i, CRLBx_3i, RMSE_3i, RMSEstdev_3i = readdata_thetaI(setnumbers_3i, subsets, iters=3)    
            
            # 3 iterations, alpha = 4
            setnumbers_4i = [222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236]
            _, xrange_4i, CRLBx_4i, RMSE_4i, RMSEstdev_4i = readdata_thetaI(setnumbers_4i, subsets, iters=3)  
    
            #VTI Theoretical
            CRLBxt_1i, xranget_1i = readdataT_thetaI(setnumbers_1i)
            CRLBxt_2i, xranget_2i = readdataT_thetaI(setnumbers_2i)
            CRLBxt_3i, xranget_3i = readdataT_thetaI(setnumbers_3i) 
            CRLBxt_4i, xranget_4i = readdataT_thetaI(setnumbers_4i)  
    
            #VTI
            ax1.loglog(xrange_1i,CRLBx_1i*10**9,color='C0')
            ax1.loglog(xrange_2i,CRLBx_2i*10**9,color='C1')
            ax1.loglog(xrange_3i,CRLBx_3i*10**9,color='C2')
            ax1.loglog(xrange_4i,CRLBx_4i*10**9,color='C3')
    
            #MAP
            ax1.errorbar(xrange_1i, RMSE_1i*10**9, yerr=RMSEstdev_1i*10**9, capsize=3, color='C0', marker='x', linestyle='None')
            ax1.errorbar(xrange_2i, RMSE_2i*10**9, yerr=RMSEstdev_2i*10**9, capsize=3, color='C1', marker='x', linestyle='None')
            ax1.errorbar(xrange_3i, RMSE_3i*10**9, yerr=RMSEstdev_3i*10**9, capsize=3, color='C2', marker='x', linestyle='None')
            ax1.errorbar(xrange_4i, RMSE_4i*10**9, yerr=RMSEstdev_4i*10**9, capsize=3, color='C3', marker='x', linestyle='None')
            
            #VTI Theoretical
            ax1.loglog(xranget_1i,CRLBxt_1i*10**9,color='C0', linestyle = 'dashed')
            ax1.loglog(xranget_2i,CRLBxt_2i*10**9,color='C1', linestyle = 'dashed')
            ax1.loglog(xranget_3i,CRLBxt_3i*10**9,color='C2', linestyle = 'dashed')
            ax1.loglog(xranget_4i,CRLBxt_4i*10**9,color='C3', linestyle = 'dashed')
            
            #Configuration
            ax1.set_xlabel(r'Expected amount of signal photons'+'\nunder illumination Eq. 4')
            ax1.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax1.grid()
            ax1.set_xlim(left=200, right = 10000)
            ax1.set_ylim(bottom = 3*10**-1, top = 30)
            ax1.tick_params(axis='both', which='major')
            ax1.tick_params(axis='both', which='minor')
        
    #################################################################################
    
    if plotnumber == "S9d":
        thetaI=2000
        
        #Opacity scale settings
        alpha1 = 0.4
        alpha2 = 2/3
        alpha3 = 1
        
        fig, (ax1) = plt.subplots(1,1)
        fig.set_size_inches(5, 5, forward=True)
        
        params0, CRLBx0, RMSE0,  RMSEstdev0, illum0, stdillum0, xrange0  = readdata(1, subsets, stepsmax)
        params1, CRLBx1, RMSE1,  RMSEstdev1, illum1, stdillum1, xrange1  = readdata(6, subsets, stepsmax)
        params2, CRLBx2, RMSE2,  RMSEstdev2, illum2, stdillum2, xrange2  = readdata(3, subsets, stepsmax)
        params3, CRLBx3, RMSE3,  RMSEstdev3, illum3, stdillum3, xrange3  = readdata(9, subsets, stepsmax)        
        
        CRLBxt0,illumt0,xranget0=readdataT(1,stepsmax)
        CRLBxt1,illumt1,xranget1=readdataT(6,stepsmax)
        CRLBxt2,illumt2,xranget2=readdataT(3,stepsmax)
        CRLBxt3,illumt3,xranget3=readdataT(9,stepsmax)
        
        if panel==1:
            #VTI
            ax1.loglog(xrange0, CRLBx0*10**9,color='C0', alpha = alpha3, marker = '^', markevery=[-1])
            
            ax1.loglog(xrange1[0:40], CRLBx1[0:40]*10**9,color='C1', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xrange1[39:80], CRLBx1[39:80]*10**9,color='C1', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xrange1[79:-1], CRLBx1[79:-1]*10**9,color='C1', alpha = alpha3, marker = 'd', markevery=[-1])
            
            ax1.loglog(xrange2[0:40], CRLBx2[0:40]*10**9,color='C2', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xrange2[39:80], CRLBx2[39:80]*10**9,color='C2', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xrange2[79:-1], CRLBx2[79:-1]*10**9,color='C2', alpha = alpha3, marker = 'd', markevery=[-1])
            
            ax1.loglog(xrange3[0:40], CRLBx3[0:40]*10**9,color='C3', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xrange3[39:80], CRLBx3[39:80]*10**9,color='C3', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xrange3[79:-1], CRLBx3[79:-1]*10**9,color='C3', alpha = alpha3, marker = 'd', markevery=[-1])
            
            #VTI Theoretical
            ax1.loglog(xranget0, CRLBxt0*10**9,color='C0', linestyle = 'dashed', alpha = alpha3, marker = '^', markevery=[-1])
            
            ax1.loglog(xranget1[0:40], CRLBxt1[0:40]*10**9,color='C1', linestyle='dashed', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xranget1[39:80], CRLBxt1[39:80]*10**9,color='C1', linestyle='dashed', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xranget1[79:-1], CRLBxt1[79:-1]*10**9,color='C1', linestyle='dashed', alpha = alpha3, marker = 'd', markevery=[-1])
            
            ax1.loglog(xranget2[0:40], CRLBxt2[0:40]*10**9,color='C2', linestyle='dashed', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xranget2[39:80], CRLBxt2[39:80]*10**9,color='C2', linestyle='dashed', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xranget2[79:-1], CRLBxt2[79:-1]*10**9,color='C2', linestyle='dashed', alpha = alpha3, marker = 'd', markevery=[-1])
            
            ax1.loglog(xranget3[0:40], CRLBxt3[0:40]*10**9,color='C3', linestyle='dashed', alpha = alpha1, marker = 'd', markevery=[-1])
            ax1.loglog(xranget3[39:80], CRLBxt3[39:80]*10**9,color='C3', linestyle='dashed', alpha = alpha2, marker = 'd', markevery=[-1])
            ax1.loglog(xranget3[79:-1], CRLBxt3[79:-1]*10**9,color='C3', linestyle='dashed', alpha = alpha3, marker = 'd', markevery=[-1])
            
            #Configuration
            ax1.set_xlabel(r'Cumulative signal photons')
            ax1.set_ylabel(r'Localization precision $\sigma_x$ [nm]')
            ax1.grid()
            ax1.set_xlim(left=thetaI/5, right = thetaI+100)
            ax1.set_ylim(bottom = 3*10**-1, top = 30)
            ax1.tick_params(axis='both', which='major')
            ax1.tick_params(axis='both', which='minor')
        
        if panel==0:
            #Legend
            black_cont = mlines.Line2D([0], [0],color='black', label=('Realistic simulation VTI' + '\n' + r'($m=0.95$, $\theta_b = 8$ ph./px.)'))
            black_dash = mlines.Line2D([0], [0],color='black', label=('RMSE of MAP estimates'), marker='x', linestyle='None')
            black_VTI = mlines.Line2D([0], [0],color='black', label=('Ideal case VTI' + '\n' +r'($m=1$, $\theta_b = 0$ ph./px.)'), linestyle='dashed')
            C0_line = mlines.Line2D([0], [0],color='C0', label=r'1 iteration (SMLM), $\phi^{\pm}_{x,0}= 0$', marker='^')
            C1_line = mlines.Line2D([0], [0],color='C1', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 2\sigma_{x,k-1}) - \pi$', marker='d')
            C2_line = mlines.Line2D([0], [0],color='C2', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 3\sigma_{x,k-1}) - \pi$', marker='d')
            C3_line = mlines.Line2D([0], [0],color='C3', label=r'3 iterations,' + '\n' + r'$\phi^{\pm}_{x,k}=\omega(\hat{\theta}_{x,k-1} \pm 4\sigma_{x,k-1}) - \pi$', marker='d')
    
            ax1.legend(handles=[black_cont, black_VTI, black_dash, C0_line, C1_line, C2_line, C3_line], prop={'size':12}, loc='upper left', bbox_to_anchor=(1.04,1), labelspacing = 1)
        

#Figure 3d
plot("3d",1)
plot("3d",2)
plot("3d",0)

#Figure 3e
plot("3e",1)
plot("3e",2)
plot("3e",0)

#Figure S9a
plot("S9a",1)

#Figure S9b
plot("S9b",1)
plot("S9b",0)

#Figure S9c
plot("S9c",1)

#Figure S9d
plot("S9d",1)
plot("S9d",0)