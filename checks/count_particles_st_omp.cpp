// @FUSE_OMP

#include "scalfmm/algorithms/omp/task_dep.hpp"
#define COUNT_USE_OPENMP

#include "count_particles_st_gen.hpp"

// examples/Release/count_particles_st_seq --input-source-file 1_source.fma --input-target-file 1_target.fma
// --tree-height 2 --dimension 2

/*
count_particles_st_omp --input-source-file ../data/sources_targets/sphere-706_source.fma   --input-target-file
../data/sources_targets/sphere-706_target.fma   --tree-height  2 --dimension 3  -t 6
*/
