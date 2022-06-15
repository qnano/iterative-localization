General information
---
This software belongs to the article:
Precision in iterative modulation enhanced single-molecule localization microscopy
DOI: https://doi.org/10.1016/j.bpj.2022.05.027

Data availability
---
The simulation data underlying the figures in the article can be found at:
DOI: https://doi.org/10.4121/19786735

Windows installation
---
The iterative localization code was developed and tested in Python v3.7.10, running on Windows 10.

Steps:

1. Clone repository or download software from https://github.com/qnano/iterative-localization. If necessary, extract the code in the directory '/iterative_localization/'

2. Install the Anaconda distribution: https://www.anaconda.com/products/individual. 

3. We recommend creating and activating a new environment for iterative_localization, as follows:

    - Open the Anaconda prompt.

    - Run the following commands in the terminal:
    ```
        conda create -n vti_env python=3.7.10 anaconda
        conda activate vti_env
    ```
    - Keep the terminal open for step 4.

4. The photonpy package (tested version 1.0.39) is needed. It can be installed as follows:
    
    - In the terminal from step 3, run:
    ```pip install photonpy==1.0.39```

5. Install CUDA Toolkit 10.1 update 2, which can be found on https://developer.nvidia.com/cuda-10.1-download-archive-update2.

Data processing
---
We describe the simulation procedure that is needed to reproduce the figures from the article. For convenience, we also provide the existing simulation data on https://doi.org/10.4121/19786735. We therefore also describe how to reproduce the figures from the article from the existing simulation data.
	
### Simulation procedure

This section describes the simulation procedure. We will go through the steps to input simulation parameters and to run the simulation software. If you intend to (re)produce results from existing simulation data, please refer to the next section.

1. Download Datasets.xlsx from https://doi.org/10.4121/19786735 and place it in the directory "/iterative_localization/". This file describes which simulation parameters are used in each simulation. It also contains an overview of which datasets are used in each of the figures in the article.

2. Verify that code and the Datasets.xlsx file are contained in the directory "/iterative_localization/".

3. Create an empty folder "/iterative_localization/SimData/". This folder will be used to store the data.

4. Open VTI_helper.py.

5. Configure the simulation parameters of the imaging setup in the function imgparams(). Specifically, you can adjust the following parameters here:
   - roisize: Amount of pixels in 1 direction of the region of interest (pixels)
   - delta_x: Pixel size of square pixels (m)
   - m: modulation contrast of the illumination patterns
   - p: Illumination pattern pitch (m)
   - L: Wavelength of emission light (m)
   - NA: Numerical aperture
   Please note that roisize, delta_x, p, L and NA are assumed to be the same for all simulations (i.e. these are not stored in the data), whereas the value of m is stored in the data. Be careful when varying roisize, delta_x, p, L and NA between simulations, as previous data will be overwritten by default.

6. Configure the ground truth estimand values in the function estimands(). Specifically, you can adjust the following values:
   - theta_x: average emitter x-position relative to first pixel, before subpixel randomization (pixels)
   - theta_y: average emitter y-position relative to first pixel, before subpixel randomization (pixels)
   - theta_I: expected signal photon intensity over all iterations under maximum illumination (photons)
   - theta_b: expected background photon count per pixel, assumed to be uniform over the region of interest (photons/pixel)
   Please note that theta_x and theta_y are assumed to be the same for all simulations (i.e. these are not stored in the data), whereas the values of theta_I and theta_b are stored in the data. Be careful when varying theta_x and theta_y between simulations, as previous data will be overwritten by default.

7. Configure the simulation parameters for iterative localization in the function VTIparams(). Specifically, you can adjust the following values:
   - MC_samples: Amount of samples used for simple Monte Carlo estimation of the VTI
   - itermax: Amount of imeSMLM iterations
   - alpha: Aggressiveness parameter, that is the amount of standard deviations between the emitter estimate and a pattern minimum
   Please note that MC_samples is assumed to be the same for all simulations (i.e. these are not stored in the data), whereas the values of of itermax and alpha are stored in the data. Be careful when varying MC_samples between simulations, as previous data will be overwritten by default.

8. Ensure that Datasets.xlsx contains an entry corresponding to the used values of itermax, alpha, theta_I, m and theta_b, as the plotting software uses this file as a reference sheet. If needed, add the entry yourself.

9. Run the appropriate simulation file(s):
   - imeSMLM_VTI.py: Simulation of the Van Trees inequality (VTI) and maximum a posteriori (MAP) estimates in imeSMLM, in the fixed photon budget scenario.
   - imeSMLM_CRLB.py: Simulation of the Cramér-Rao lower bound (CRLB) and the quadratic approximation of the CRLB computed over all iterations in imeSMLM, in the fixed photon budget scenario.
   - imeSMLM_TheoreticalVTI.py: Simulation of the analytical approximation of the VTI in imeSMLM, in the fixed photon budget scenario.
   - FT_imeSMLM_VTI.py: Simulation of the Van Trees inequality (VTI) and maximum a posteriori (MAP) estimates in imeSMLM, in the fixed imaging time and fixed illumination intensity scenario.
   - FT_imeSMLM_TheoreticalVTI.py: Simulation of the analytical approximation of the VTI in imeSMLM, in the fixed imaging time and fixed illumination intensity scenario.

10. The simulation data will be saved in the directory "/iterative_localization/SimData/".

11. To do additional simulations, repeat steps 3-7 to configure the simulation parameters and steps 8, 9 to run the simulations.

### Using existing data

This section describes how to use the existing simulation data from https://doi.org/10.4121/19786735. If you intend to (re)produce results by running the simulation yourself, please refer to the previous section.

1. Download Datasets.xlsx and place it in the directory "/iterative_localization/". This file describes which simulation parameters are used in each simulation. It also contains an overview of which datasets are used in each of the figures in the article.

2. Download the appropriate (raw and processed) simulation data from https://doi.org/10.4121/19786735. Specifically:

    - If you intend to only reproduce the results and want to use the minimum amount of data to do so, download the following data and extract it to the directory "/iterative_localization/SimData/":
	* CRLBxI.zip: Cramér-Rao lower bound, computed over all iterations in the fixed photon budget scenario.
	* CRLBxM.zip: Approximation of the Cramér-Rao lower bound in the fixed photon budget scenario, from Balzarotti et al. (2017). Nanometer resolution imaging and tracking of fluorescent molecules with minimal photon fluxes. Science 355:606–612. https://www.science.org/doi/10.1126/science.aak9913.
	* FT-J.zip: Bayesian information matrix in the fixed imaging time and fixed illumination intensity scenario.
	* FT-mod.zip: Fraction of the photon budget used in each iteration in the fixed imaging time and fixed illumination intensity scenario.
	* FT-modT.zip: Fraction of the photon budget used in each iteration assuming the analytical approximation of the VTI, in the fixed imaging time and fixed illumination intensity scenario.
	* FT-smp.zip: Regions of interest in the fixed imaging time and fixed illumination intensity scenario.
	* FT-theta.zip: Ground truth estimand values after subpixel randomization in the fixed imaging time and fixed illumination intensity scenario.
	* FT-thetaMAP.zip: Maximum a posteriori (MAP) estimated parameter values in the fixed imaging time and fixed illumination intensity scenario.
	* FT-VTIx.zip: Van Trees inequality (VTI) in the fixed imaging time and fixed illumination intensity scenario.
	* FT-VTIxT.zip: Analytical approximation of the VTI in the fixed imaging time and fixed illumination intensity scenario.
	* J.zip: Bayesian information matrix in the fixed photon budget scenario.
	* smp.zip: Regions of interest in the fixed photon budget scenario.
	* theta.zip: Ground truth estimand values after subpixel randomization in the fixed photon budget scenario.
	* thetaMAP.zip: Maximum a posteriori (MAP) estimated parameter values in the fixed photon budget scenario.
	* VTIx.zip: Van Trees inequality (VTI) in the fixed photon budget scenario.
	* VTIxT.zip: Analytical approximation of the VTI in the fixed photon budget scenario.

    - If you want to access all the data, including (raw) quantities that are not directly used to compute the results shown in the article, download SimData.zip and extract it to the directory "/iterative_localization/SimData/". SimData.zip contains all data from individual .zip-files, combined for your convenience. Alternatively, you can download all individual .zip-files and extract them to the directory "/iterative_localization/SimData/".

### Reproducing results

1. Verify that the directory "iterative_localization/SimData/" exists and that it contains the necessary and desired simulation data. Additionally, verify that code and the Datasets.xlsx file are contained in the directory "/iterative_localization/".

2. To produce the desired figures from the article, open the appropriate plotting file(s):
   - plot_precision.py: Figures 2d, 2e, S2a, S2b, S5, S6a, S6b, S6c, S7a, S7b, S8a, S8b, S8c, S10a, S10b, S11a, S11b
   - FT_plot_precision.py: Figures 3d, 3e, S9a, S9b, S9c, S9d
   - plot_samples.py: Figures 2a, 2b, 2c, S4
   - FT_plot_samples.py: Figures 3a, 3b, 3c

3. Run the appropriate file(s) to process and plot the data. By default, this will produce all individual figure panels and legends.

4. To display specific figures or panels, you can use the function plot(plotnumber, panel) after running the file once. To do so:
   - Input the figure number as a string.
   - In plot_precision.py and FT_plot_precision.py, also input the panel number. Use 1 for the left panel, 2 for the right panel and 0 for the legend.
   - As an example, to display the right panel of Figure 2d, use plot("2d", 2) after running plot_precision.py.

Description of individual Python files
---
### VTI_helper.py
Module of functions for iterative meSMLM VTI computation

   - imgparams() reports values for parameters related to the imaging system
   - estimands() reports values for to-be-estimated parameters
   - VTIparams() reports values for parameters needed to evaluate the VTI
   - intensity(x, phi) returns the illumination intensity of a sinusoidal pattern for a given coordinate and phase shift
   - MC_sample(mean, cov) returns sampled parameter values of the simple MC-sampling distibution (here, a Gaussian)
   - FIM(mod) returns the Fisher information matrix
   - VTI_prior(mean, cov, samples) returns the Bayesian prior information matrix, assuming a Gaussian prior
   - VTI_data(mod, samples) returns the Bayesian data information matrix
   - VTI_eval(mean, cov, mod) returns a simple Monte Carlo estimate of the Van Trees inequality, given prior information mean and cov and imaging configuration contained in mod
   - VTI_theoretical(iteration, mod, theta_x, theta_prev=None, sigma_prev=None) returns the theoretical (analytical approximation) Van Trees inequality after each iteration, given an imaging configuration contained in mod
   - MAP_eval(mod_per_spot) returns an array of maximum a posteriori estimates under given imaging conditions, as well as the Cramér-Rao lower bound and the Fisher information matrix

### FT_imeSMLM_TheoreticalVTI.py
Simulation of the analytical approximation of the VTI in imeSMLM, in the fixed 
imaging time and fixed illumination intensity scenario.

In this version of the code, parameters c_k are chosen such that the imaging 
time and illumination intensity are fixed per iteration. As such, the amount of 
photons that are recorded during an iteration is not fixed in simulation.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate VTI 
values between the endpoints of iterations.

### FT_imeSMLM_VTI.py
Simulation of the Van Trees inequality (VTI) and maximum a posteriori (MAP) 
estimates in imeSMLM, in the fixed imaging time and fixed illumination intensity 
scenario.

In this version of the code, parameters c_k are chosen such that the imaging 
time and illumination intensity are fixed per iteration. As such, the amount of 
photons that are recorded during an iteration is not fixed in simulation.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate VTI 
values between the endpoints of iterations.

### imeSMLM_CRLB.py
Simulation of the Cramér-Rao lower bound (CRLB) and the quadratic approximation 
of the CRLB computed over all iterations in imeSMLM, in the fixed photon budget 
scenario.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate CRLB 
values between the endpoints of iterations.

### imeSMLM_TheoreticalVTI.py
Simulation of the analytical approximation of the VTI in imeSMLM, in the fixed 
photon budget scenario.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate VTI 
values between the endpoints of iterations.

### imeSMLM_VTI.py
Simulation of the Van Trees inequality (VTI) and maximum a posteriori (MAP) 
estimates in imeSMLM, in the fixed photon budget scenario.

Uses photonpy for CUDA parallelization, and evaluates stepsmax intermediate VTI 
values between the endpoints of iterations.

### FT_plot_precision.py
Reading data, processing and plotting of Figures
   - 3d, 3e, 
   - S9a, S9b, S9c, S9d

### FT_plot_samples.py
Reading data, processing and plotting of Figures
   - 3a, 3b, 3c

### plot_precision.py
Reading data, processing and plotting of Figures
   - 2d, 2e, 
   - S2a, S2b, 
   - S5, 
   - S6a, S6b, S6c, 
   - S7a, S7b, 
   - S8a, S8b, S8c, 
   - S10a, S10b, 
   - S11a, S11b

### plot_samples.py
Reading data, processing and plotting of Figures
   - 2a, 2b, 2c
   - S4