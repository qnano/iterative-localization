# -*- coding: utf-8 -*-
"""
Simulation of the Van Trees inequality (VTI) and maximum a posteriori (MAP) 
estimates in imeSMLM, in the fixed photon budget scenario.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate VTI 
values between the endpoints of iterations.
"""

import numpy as np
import VTI_helper as vti

#Path to save data to
savepath = './SimData/'

#Set random seed
np.random.seed(0)

#Amount of MAP-runs per iteration
maxruns = 50000;

#Load parameters
roisize,m,kx,ky,sigmap,other = vti.imgparams();
delta_x = other[0];

#Subpixel randomization of ground truth positions
theta = vti.estimands();
theta_per_spot=np.repeat(theta,maxruns,0)
theta_per_spot[:,[0,1]] += np.random.uniform(-1/2,1/2,size=(maxruns,2))

MC_samples,itermax,alpha = vti.VTIparams();

#Configure the amount of intermediate plotting steps per iteration
stepsmax = int(120/itermax);

#Initialize matrices
phi_plusx_id = np.zeros((itermax))
phi_minusx_id = np.zeros((itermax))
phi_plusy_id = np.zeros((itermax))
phi_minusy_id = np.zeros((itermax))

I_plusx_id = np.zeros((itermax))
I_minusx_id = np.zeros((itermax))
I_plusy_id = np.zeros((itermax))
I_minusy_id = np.zeros((itermax))
I_sum_id = np.zeros((itermax))

phi_plusx = np.zeros((maxruns,itermax))
phi_minusx = np.zeros((maxruns,itermax))
phi_plusy = np.zeros((maxruns,itermax))
phi_minusy = np.zeros((maxruns,itermax))

I_plusx = np.zeros((maxruns,itermax))
I_minusx = np.zeros((maxruns,itermax))
I_plusy = np.zeros((maxruns,itermax))
I_minusy = np.zeros((maxruns,itermax))
I_sum = np.zeros((maxruns,itermax))

J = np.zeros((itermax,np.size(theta),np.size(theta)))
J_inv = np.zeros((itermax,np.size(theta),np.size(theta)))
Jsmooth = np.zeros((itermax*stepsmax,np.size(theta),np.size(theta)))
Jsmooth_inv = np.zeros((itermax*stepsmax,np.size(theta),np.size(theta)))

Jt = np.zeros((itermax))
Jt_inv = np.zeros((itermax))
Jtsmooth = np.zeros((itermax*stepsmax))
Jtsmooth_inv = np.zeros((itermax*stepsmax))

thetaMAP=np.zeros((maxruns,itermax,np.size(theta)))
CRLB = np.zeros((maxruns,itermax,np.size(theta)))
I = np.zeros((maxruns,itermax,np.size(theta),np.size(theta)))

mod_storage = np.zeros((maxruns,itermax*4, 6))

for k in range(itermax):
    
# =============================================================================
#                               Iteration 0
# =============================================================================
    if k == 0:
        print('Iteration 0 started. Evaluating VTI...')
        #Evaluate the Fisher information with 4 patterns
        phi_plusx[:,k]  =   kx*theta[0,0] - np.pi/2; 
        phi_minusx[:,k] =   kx*theta[0,0] - np.pi/2;
        phi_plusy[:,k]  =   ky*theta[0,1] - np.pi/2; 
        phi_minusy[:,k] =   ky*theta[0,1] - np.pi/2; 
           
        I_plusx_id[k]    =   vti.intensity(theta[0,0],phi_plusx[0,k]);
        I_minusx_id[k]   =   vti.intensity(theta[0,0],phi_minusx[0,k]);
        I_plusy_id[k]    =   vti.intensity(theta[0,1],phi_plusy[0,k]);
        I_minusy_id[k]   =   vti.intensity(theta[0,1],phi_minusy[0,k]);
        I_sum_id[k]      =   I_plusx_id[k] + I_minusx_id[k] + I_plusy_id[k] + I_minusy_id[k];
        
# =============================================================================
#                                   VTI
# =============================================================================
        for step in range(stepsmax):
            #Compute the fraction of photons used until this part of the iteration
            fraction = (step+1)/stepsmax;
            
            mod = np.array([
            #kx, ky, kz, modulation contrast, phase shift of sin(omega*x-phi) (maximum at pi/2, minimum at -pi/2), relative intensity
                [kx, 0,  0,  m, phi_plusx[0,k],   fraction/(itermax*I_sum_id[k])], 
                [kx, 0,  0,  m, phi_minusx[0,k],   fraction/(itermax*I_sum_id[k])], 
                [0,  ky, 0,  m, phi_plusy[0,k],   fraction/(itermax*I_sum_id[k])],  
                [0,  ky, 0,  m, phi_minusy[0,k],   fraction/(itermax*I_sum_id[k])], 
            ])
            
            Jsmooth[k*stepsmax+step,:,:] = vti.FIM(mod);
            Jsmooth_inv[k*stepsmax+step,:,:] = np.linalg.inv(Jsmooth[k*stepsmax+step,:,:]);
            
            if step+1==stepsmax:
                J[k,:,:]=Jsmooth[k*stepsmax+step,:,:];
                J_inv[k,:,:]=Jsmooth_inv[k*stepsmax+step,:,:];
                
# =============================================================================
#                   MAP: only on final step of iteration
# =============================================================================
        print('Evaluating MAP...')
        for runs in range(maxruns):
            I_plusx[runs,k]    =   vti.intensity(theta_per_spot[runs,0],phi_plusx[runs,k]);
            I_minusx[runs,k]   =   vti.intensity(theta_per_spot[runs,0],phi_minusx[runs,k]);
            I_plusy[runs,k]    =   vti.intensity(theta_per_spot[runs,1],phi_plusy[runs,k]);
            I_minusy[runs,k]   =   vti.intensity(theta_per_spot[runs,1],phi_minusy[runs,k]);
            I_sum[runs,k]      =   I_plusx[runs,k] + I_minusx[runs,k] + I_plusy[runs,k] + I_minusy[runs,k];

        mod = np.array([
        #kx, ky, kz, modulation contrast, phase shift of sin(omega*x-phi) (maximum at pi/2, minimum at -pi/2), relative intensity
            [kx, 0,  0,  m, phi_plusx[runs,k],   1/(itermax*I_sum[runs,k])], 
            [kx, 0,  0,  m, phi_minusx[runs,k],   1/(itermax*I_sum[runs,k])], 
            [0,  ky, 0,  m, phi_plusy[runs,k],   1/(itermax*I_sum[runs,k])],  
            [0,  ky, 0,  m, phi_minusy[runs,k],   1/(itermax*I_sum[runs,k])], 
        ])

        #Compute maximum a posteriori estimate for maxruns spots
        mod_storage[:,4*k:4*(k+1),:] = mod;
        mod_per_spot = mod_storage[:,0:4*(k+1),:]
        estim,crlb,fi,smp=vti.MAP_eval(mod_per_spot, theta_per_spot, rejectUnconverged=True);
                                    
        #Store MAP
        thetaMAP[:,k,:]=  estim
        CRLB[:,k,:]=      crlb
        I[:,k,:,:]=       fi

# =============================================================================
#                               Iterations 1+
# =============================================================================
    elif k > 0 and k<itermax:
        print('Iteration ' + str(k) + ' started. Evaluating VTI...')
        #Initialize imaging condition matrix
        mod_per_spot=np.zeros((maxruns,4*(k+1),6));

# =============================================================================
#                                   VTI
# =============================================================================  
        #Evaluate the Bayesian information with 4 ideally positioned patterns
        phi_plusx_id[k]  =   kx*theta[0,0] + np.pi/2 +   alpha * kx*(J_inv[k-1,0,0])**0.5; 
        phi_minusx_id[k] =   kx*theta[0,0] + np.pi/2 -   alpha * kx*(J_inv[k-1,0,0])**0.5;
        phi_plusy_id[k]  =   ky*theta[0,1] + np.pi/2 +   alpha * kx*(J_inv[k-1,1,1])**0.5; 
        phi_minusy_id[k] =   ky*theta[0,1] + np.pi/2 -   alpha * kx*(J_inv[k-1,1,1])**0.5; 
        
        I_plusx_id[k]    =   vti.intensity(theta[0,0],phi_plusx_id[k]);
        I_minusx_id[k]   =   vti.intensity(theta[0,0],phi_minusx_id[k]);
        I_plusy_id[k]    =   vti.intensity(theta[0,1],phi_plusy_id[k]);
        I_minusy_id[k]   =   vti.intensity(theta[0,1],phi_minusy_id[k]);
        I_sum_id[k]      =   I_plusx_id[k] + I_minusx_id[k] + I_plusy_id[k] + I_minusy_id[k];
     
        for step in range(stepsmax):
            #Compute the fraction of photons used until this part of the iteration
            fraction = (step+1)/stepsmax;
            
            mod = np.array([
            #kx, ky, kz, modulation contrast, phase shift of sin(x-phi) (maximum at pi/2, minimum at -pi/2), relative intensity
            [kx, 0,  0,  m, phi_plusx_id[k],     fraction/(itermax*I_sum_id[k])],
            [kx, 0,  0,  m, phi_minusx_id[k],    fraction/(itermax*I_sum_id[k])],
            [0,  ky, 0,  m, phi_plusy_id[k],     fraction/(itermax*I_sum_id[k])],
            [0,  ky, 0,  m, phi_minusy_id[k],    fraction/(itermax*I_sum_id[k])],
            ])
            
            Jsmooth[k*stepsmax+step,:,:] = vti.VTI_eval(theta,J_inv[k-1,:,:],mod,bgcompensation=True);
            Jsmooth_inv[k*stepsmax+step,:,:] = np.linalg.inv(Jsmooth[k*stepsmax+step,:,:]);
            
            if step+1==stepsmax:
                J[k,:,:]=Jsmooth[k*stepsmax+step,:,:];
                J_inv[k,:,:]=Jsmooth_inv[k*stepsmax+step,:,:];

# =============================================================================
#                   MAP: only on final step of iteration
# =============================================================================
        
        for runs in range(maxruns):
            #Evaluate the MAP with 4 patterns
            phi_plusx[runs,k]  =   kx*thetaMAP[runs,k-1,0] + np.pi/2 +   alpha * kx*(J_inv[k-1,0,0])**0.5; 
            phi_minusx[runs,k] =   kx*thetaMAP[runs,k-1,0] + np.pi/2 -   alpha * kx*(J_inv[k-1,0,0])**0.5;
            phi_plusy[runs,k]  =   ky*thetaMAP[runs,k-1,1] + np.pi/2 +   alpha * kx*(J_inv[k-1,1,1])**0.5; 
            phi_minusy[runs,k] =   ky*thetaMAP[runs,k-1,1] + np.pi/2 -   alpha * kx*(J_inv[k-1,1,1])**0.5; 
                
            I_plusx[runs,k]    =   vti.intensity(theta_per_spot[runs,0],phi_plusx[runs,k]);
            I_minusx[runs,k]   =   vti.intensity(theta_per_spot[runs,0],phi_minusx[runs,k]);
            I_plusy[runs,k]    =   vti.intensity(theta_per_spot[runs,1],phi_plusy[runs,k]);
            I_minusy[runs,k]   =   vti.intensity(theta_per_spot[runs,1],phi_minusy[runs,k]);
            I_sum[runs,k]      =   I_plusx[runs,k] + I_minusx[runs,k] + I_plusy[runs,k] + I_minusy[runs,k];
            
            mod= np.array([
            #kx, ky, kz, modulation contrast, phase shift of sin(x-phi) (maximum at pi/2, minimum at -pi/2), relative intensity
            [kx, 0,  0,  m, phi_plusx[runs,k],     1/(itermax*I_sum[runs,k])],
            [kx, 0,  0,  m, phi_minusx[runs,k],    1/(itermax*I_sum[runs,k])],
            [0,  ky, 0,  m, phi_plusy[runs,k],     1/(itermax*I_sum[runs,k])],
            [0,  ky, 0,  m, phi_minusy[runs,k],    1/(itermax*I_sum[runs,k])],
            ])
            
            mod_storage[runs,4*k:4*(k+1),:] = mod;
            mod_per_spot[runs,:,:]=mod_storage[runs,0:4*(k+1),:];
            
        #Compute MAP
        print('Evaluating MAP...')
        estim,crlb,fi,smp = vti.MAP_eval(mod_per_spot,theta_per_spot,previousData=smp,rejectUnconverged=True);        
       
        #Store MAP
        thetaMAP[:,k,:]=  estim
        CRLB[:,k,:]=      crlb
        I[:,k,:,:]=       fi

CRLBx = np.sqrt(Jsmooth_inv[:,0,0])*delta_x;
CRLBy = np.sqrt(Jsmooth_inv[:,1,1])*delta_x;
CRLBI = np.sqrt(Jsmooth_inv[:,2,2]);
CRLBb = np.sqrt(Jsmooth_inv[:,3,3]);

# =============================================================================
#                       Save data
# =============================================================================
saveCRLB = 'VTIx-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))
saveI = 'I-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))
saveJ = 'J-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))
savethetaMAP = 'thetaMAP-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))
savesmp ='smp-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))
savetheta = 'theta-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))

np.save(savepath + saveCRLB, CRLBx)
np.save(savepath + saveI, I)
np.save(savepath + saveJ, J)
np.save(savepath + savethetaMAP, thetaMAP)
np.save(savepath + savesmp, smp)
np.save(savepath + savetheta,theta_per_spot)