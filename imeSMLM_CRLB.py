# -*- coding: utf-8 -*-
"""
Simulation of the Cramér-Rao lower bound (CRLB) and the quadratic approximation 
of the CRLB computed over all iterations in imeSMLM, in the fixed photon budget 
scenario.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate CRLB 
values between the endpoints of iterations.
"""

import numpy as np
import VTI_helper as vti

#Path to save data to
savepath = './SimData/'

#Set random seed
np.random.seed(0)

#Load parameters
roisize,m,kx,ky,sigmap,other = vti.imgparams();
delta_x = other[0];
theta = vti.estimands();
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

phi_plusx = np.zeros((1,itermax))
phi_minusx = np.zeros((1,itermax))
phi_plusy = np.zeros((1,itermax))
phi_minusy = np.zeros((1,itermax))

I_plusx = np.zeros((1,itermax))
I_minusx = np.zeros((1,itermax))
I_plusy = np.zeros((1,itermax))
I_minusy = np.zeros((1,itermax))
I_sum = np.zeros((1,itermax))

J = np.zeros((itermax,np.size(theta),np.size(theta)))
J_inv = np.zeros((itermax,np.size(theta),np.size(theta)))
Jsmooth = np.zeros((itermax*stepsmax,np.size(theta),np.size(theta)))
Jsmooth_inv = np.zeros((itermax*stepsmax,np.size(theta),np.size(theta)))
CRLB_minflux = np.zeros(itermax*stepsmax)

Jt = np.zeros((itermax))
Jt_inv = np.zeros((itermax))
Jtsmooth = np.zeros((itermax*stepsmax))
Jtsmooth_inv = np.zeros((itermax*stepsmax))

thetaMLE=np.zeros((1,itermax,np.size(theta)))
CRLB = np.zeros((1,itermax,np.size(theta)))
I = np.zeros((1,itermax,np.size(theta),np.size(theta)))

mod_storage = np.zeros((1,itermax*4, 6))

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
            
        I_plusx[:,k]    =   vti.intensity(theta[0,0],phi_plusx[0,k]);
        I_minusx[:,k]   =   vti.intensity(theta[0,0],phi_minusx[0,k]);
        I_plusy[:,k]    =   vti.intensity(theta[0,1],phi_plusy[0,k]);
        I_minusy[:,k]   =   vti.intensity(theta[0,1],phi_minusy[0,k]);
        I_sum[:,k]      =   I_plusx[:,k] + I_minusx[:,k] + I_plusy[:,k] + I_minusy[:,k];
        
# =============================================================================
#                                   VTI
# =============================================================================
        for step in range(stepsmax):
            #Compute the fraction of photons used until this part of the iteration
            fraction = (step+1)/stepsmax;
            
            mod = np.array([
            #kx, ky, kz, modulation contrast, phase shift of sin(omega*x-phi) (maximum at pi/2, minimum at -pi/2), relative intensity
                [kx, 0,  0,  m, phi_plusx[0,k],   fraction/(itermax*I_sum[0,k])], 
                [kx, 0,  0,  m, phi_minusx[0,k],   fraction/(itermax*I_sum[0,k])], 
                [0,  ky, 0,  m, phi_plusy[0,k],   fraction/(itermax*I_sum[0,k])],  
                [0,  ky, 0,  m, phi_minusy[0,k],   fraction/(itermax*I_sum[0,k])], 
            ])
            
            Jsmooth[k*stepsmax+step,:,:] = vti.FIM(mod);
            Jsmooth_inv[k*stepsmax+step,:,:] = np.linalg.inv(Jsmooth[k*stepsmax+step,:,:]);
            
            Lk = 1/other[2]*2*np.pi
            CRLB_minflux[k*stepsmax+step] = Lk/(4*np.sqrt(fraction*theta[0,2]/itermax))
            
            if step+1==stepsmax:
                mod_storage[:,4*k:4*(k+1),:] = mod;
                J[k,:,:]=Jsmooth[k*stepsmax+step,:,:];
                J_inv[k,:,:]=Jsmooth_inv[k*stepsmax+step,:,:];

# =============================================================================
#                               Iterations 1+
# =============================================================================
    elif k > 0 and k<itermax:
        print('Iteration ' + str(k) + ' started. Evaluating VTI...')
        #Initialize imaging condition matrix
        mod_current=np.zeros((4*(k+1),6));

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
            
            mod_storage[:,4*k:4*(k+1),:] = mod;
            mod_current[:,:]=mod_storage[:,0:4*(k+1),:];
            
            Jsmooth[k*stepsmax+step,:,:] = vti.FIM(mod_current);
            Jsmooth_inv[k*stepsmax+step,:,:] = np.linalg.inv(Jsmooth[k*stepsmax+step,:,:]);
            
            Lk = alpha*CRLB_minflux[k*stepsmax-1]
            CRLB_minflux[k*stepsmax+step] = Lk/(4*np.sqrt(fraction*theta[0,2]/itermax))
            
            if step+1==stepsmax:
                J[k,:,:]=Jsmooth[k*stepsmax+step,:,:];
                J_inv[k,:,:]=Jsmooth_inv[k*stepsmax+step,:,:];
                
CRLBx = np.sqrt(Jsmooth_inv[:,0,0])*delta_x;
CRLBy = np.sqrt(Jsmooth_inv[:,1,1])*delta_x;
CRLBI = np.sqrt(Jsmooth_inv[:,2,2]);
CRLBb = np.sqrt(Jsmooth_inv[:,3,3]);

# =============================================================================
#                       Save data
# =============================================================================
saveCRLB = 'CRLBxI-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))
saveCRLBM = 'CRLBxM-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))

np.save(savepath + saveCRLB, CRLBx)
np.save(savepath + saveCRLBM, CRLB_minflux)