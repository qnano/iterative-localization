# -*- coding: utf-8 -*-
"""
Simulation of the analytical approximation of the VTI in imeSMLM, in the fixed 
photon budget scenario.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate VTI 
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
theta[0,3] = 0;
_,itermax,alpha = vti.VTIparams();

#Configure the amount of intermediate plotting steps per iteration
stepsmax = int(120/itermax);

if m != 1:
    print('Theoretical approximation of the VTI is evaluated for m = 1.')

#Initialize matrices
phi_plusx = np.zeros((itermax))
phi_minusx = np.zeros((itermax))
phi_plusy = np.zeros((itermax))
phi_minusy = np.zeros((itermax))

I_plusx = np.zeros((itermax))
I_minusx = np.zeros((itermax))
I_plusy = np.zeros((itermax))
I_minusy = np.zeros((itermax))
I_sum = np.zeros((itermax))

Jt = np.zeros((itermax))
Jt_inv = np.zeros((itermax))
Jtsmooth = np.zeros((itermax*stepsmax))
Jtsmooth_inv = np.zeros((itermax*stepsmax))

for k in range(itermax):
    
# =============================================================================
#                               Iteration 0
# =============================================================================
    if k == 0:
        print('Iteration 0 started. Evaluating VTI...')
        #Evaluate the Fisher information with 4 patterns
        phi_plusx[k]  =   kx*theta[0,0] - np.pi/2; 
        phi_minusx[k] =   kx*theta[0,0] - np.pi/2;
        phi_plusy[k]  =   ky*theta[0,1] - np.pi/2; 
        phi_minusy[k] =   ky*theta[0,1] - np.pi/2; 
            
        I_plusx[k]    =   vti.intensity(theta[0,0],phi_plusx[k]);
        I_minusx[k]   =   vti.intensity(theta[0,0],phi_minusx[k]);
        I_plusy[k]    =   vti.intensity(theta[0,1],phi_plusy[k]);
        I_minusy[k]   =   vti.intensity(theta[0,1],phi_minusy[k]);
        I_sum[k]      =   I_plusx[k] + I_minusx[k] + I_plusy[k] + I_minusy[k];
        
# =============================================================================
#                                   VTI
# =============================================================================
        for step in range(stepsmax):
            #Compute the fraction of photons used until this part of the iteration
            fraction = (step+1)/stepsmax;
            
            mod = np.array([
            #kx, ky, kz, modulation contrast, phase shift of sin(omega*x-phi) (maximum at pi/2, minimum at -pi/2), relative intensity
                [kx, 0,  0,  m, phi_plusx[k],   fraction/(itermax*I_sum[k])], 
                [kx, 0,  0,  m, phi_minusx[k],   fraction/(itermax*I_sum[k])], 
                [0,  ky, 0,  m, phi_plusy[k],   fraction/(itermax*I_sum[k])],  
                [0,  ky, 0,  m, phi_minusy[k],   fraction/(itermax*I_sum[k])], 
            ])
            
            #Theoretical VTI
            Jtsmooth[k*stepsmax+step] = vti.VTI_theoretical(k,mod)
            Jtsmooth_inv[k*stepsmax+step] = 1/(Jtsmooth[k*stepsmax+step]);            
            
            if step+1==stepsmax:
                Jt[k]=Jtsmooth[k*stepsmax+step];
                Jt_inv[k]=Jtsmooth_inv[k*stepsmax+step];
                

# =============================================================================
#                               Iterations 1+
# =============================================================================
    elif k > 0 and k<itermax:
        print('Iteration ' + str(k) + ' started. Evaluating VTI...')

# =============================================================================
#                                   VTI
# =============================================================================  
        #Evaluate the Bayesian information with 4 ideally positioned patterns
        phi_plusx[k]  =   kx*theta[0,0] + np.pi/2 +   alpha * kx*(Jt_inv[k-1])**0.5; 
        phi_minusx[k] =   kx*theta[0,0] + np.pi/2 -   alpha * kx*(Jt_inv[k-1])**0.5;
        phi_plusy[k]  =   ky*theta[0,1] + np.pi/2 +   alpha * kx*(Jt_inv[k-1])**0.5; 
        phi_minusy[k] =   ky*theta[0,1] + np.pi/2 -   alpha * kx*(Jt_inv[k-1])**0.5; 
        
        I_plusx[k]    =   vti.intensity(theta[0,0],phi_plusx[k]);
        I_minusx[k]   =   vti.intensity(theta[0,0],phi_minusx[k]);
        I_plusy[k]    =   vti.intensity(theta[0,1],phi_plusy[k]);
        I_minusy[k]   =   vti.intensity(theta[0,1],phi_minusy[k]);
        I_sum[k]      =   I_plusx[k] + I_minusx[k] + I_plusy[k] + I_minusy[k];
     
        for step in range(stepsmax):
            #Compute the fraction of photons used until this part of the iteration
            fraction = (step+1)/stepsmax;
            
            mod = np.array([
            #kx, ky, kz, modulation contrast, phase shift of sin(x-phi) (maximum at pi/2, minimum at -pi/2), relative intensity
            [kx, 0,  0,  m, phi_plusx[k],     fraction/(itermax*I_sum[k])],
            [kx, 0,  0,  m, phi_minusx[k],    fraction/(itermax*I_sum[k])],
            [0,  ky, 0,  m, phi_plusy[k],     fraction/(itermax*I_sum[k])],
            [0,  ky, 0,  m, phi_minusy[k],    fraction/(itermax*I_sum[k])],
            ])
            
            #Theoretical VTI
            Jtsmooth[k*stepsmax+step] = vti.VTI_theoretical(k,mod,theta[0,0],Jt_inv[k-1]**0.5)
            Jtsmooth_inv[k*stepsmax+step] = 1/(Jtsmooth[k*stepsmax+step]);    
            
            if step+1==stepsmax:
                Jt[k]=Jtsmooth[k*stepsmax+step];
                Jt_inv[k]=Jtsmooth_inv[k*stepsmax+step]; 

CRLBxt = np.sqrt(Jtsmooth_inv)*delta_x;

# =============================================================================
#                       Save data
# =============================================================================
saveCRLB = 'VTIxT-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))
saveJ = 'JT-iter-' + str(itermax) + '-thetaI-' + str(int(theta[0,2])) + '-m-' + str(int(100*m)) + '-alpha-' + str(int(100*alpha)) + '-thetab-' + str(int(theta[0,3]))

np.save(savepath + saveCRLB, CRLBxt)
np.save(savepath + saveJ, Jt)