Exemples 


Files 

generic 
- `tutorial.cpp`  (This code is explained in the QUICKSTART.md file)
- `test_laplace_kernels.cpp` presents the different Laplace kernels available in ScalFMM
- `test_particles.cpp` shows how particles work
- `fmm_source_target` presents a source-target simulation 

-  `test_dimension.cpp`
-  `test_dimension_omp.cpp`
-  `test_dimension_low_rank.cpp`

Particle counting codes use a specific kernel to count the particles in the simulation box.
- `count_particles_seq.cpp`	 sequential version 
- `count_particles_omp.cpp`	 OpenMP version 
- `count_particles_st_seq.cpp` source-target sequential version
- `count_particles_st_omp.cpp` source-target OpenMP version


To check
- `test_periodic_dist`
- `test_time_loop.cpp`
- `test_tensorial_interpolator.cpp`
- `test_like_mrhs.cpp`


