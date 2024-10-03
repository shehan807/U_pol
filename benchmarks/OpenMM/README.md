# Notes
* Herein, folders contain OpenMM realizations of computing the electrostatic and polarization energies
* Thole screening is only for intra-molecular dipole-dipole interactions (i.e., the water model, SWM4-NDP, has only one drude particle such that there are no Thole screening interactions)
* The acetonitrile case has 3 drude sites (C, C, N) and is an application of Thole screening for intra-molecular induced-dipole/induced-dipole Coulomb interactions 
* note how the water particle masses are zeroed out in the .xml file; this is needed to call''simulation.step(1)'' to optimize Drude positions with ''DrudeSCFIntegrator''
* OpenMM automatically excludes intra-molecular, static charge Coulomb interactions (1-4) by its bond topology--for now, intermoleclar interaction energy calculations omit intra-molecular, static charge Coulomb interactions 
* CPU kernel may experience some convergence issues resolved by switching to the CUDA kernel 

