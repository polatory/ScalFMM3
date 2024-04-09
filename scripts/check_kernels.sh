#!/bin/bash
source ${HOME}/Config/bashrc.bash
gnu
dirtools=./tools/RelWithDebInfo
direxa=./examples/RelWithDebInfo
dim=3
N=500
data_file=cube_500.bfma 
direct_result="cube_500_ref.bfma"
index=0,1,2
# Matrix kernels:
#        0 1/r, 1) grad(1/r), 2) p & grad(1/r) 3) mrhs,
#       4) 1/r^2  5) ln in 2d
kernel=1 
fmm_result="cube_500_fmm.bfma"
#The interpolation : 0 for uniform, 1 for chebyshev.
interpolator=0
echo "./tools/RelWithDebInfo/generate_distribution --n ${N} --in_val 1 --dim 3 --output-file ${data_file} --cuboid 2:2:2 --sort"
${dirtools}/generate_distribution --n ${N} --in_val 1 --dim ${dim} --output-file ${data_file} --cuboid 2:2:2 --sort --use-float


#build direct computation
echo "${dirtools}/direct_computation  --input-file ${data_file} --kernel ${kernel} --dimension ${dim} --output-file ${direct_result}
"
${dirtools}/direct_computation  --input-file ${data_file} --kernel ${kernel} --dimension ${dim} --output-file ${direct_result}
#
# FMM computation 
# 
echo "FMM computation"
echo "${direxa}/test_laplace_kernels --order 6 --h 3 --kernel ${kernel} --interpolator ${interpolator} --output-file ${fmm_result}
"
${direxa}/test_laplace_kernels --order 6 --tree-height 4 --kernel ${kernel} --interpolator ${interpolator} --input-file ${data_file} --output-file ${fmm_result} --data-sorted
#
#
echo "${dirtools}/compare_files --input-file1 ${direct_result}  --input-file2 ${fmm_result} --index1 ${index}"
