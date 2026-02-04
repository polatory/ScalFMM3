#!/bin/bash
# Read a string with spaces using for loop

files=$(find ./include/ -type f)

for value in $files
do
    /home/p13ro/dev/cpp/github/include-what-you-use/build/bin/include-what-you-use  -I ./include -I ./modules/internal/xsimd/include -I ./modules/internal/xtensor/include -I ./modules/internal/xtl/include -I ./modules/internal/inria_tools/ -I ./modules/internal/xtensor-blas/include -I ./modules/internal/xtensor-fftw -I ./modules/internal/cpp_tools/cl_parser/include -I ./modules/internal/cpp_tools/colors/include/ -I ./modules/internal/cpp_tools/parallel_manager/include/ -I ./modules/internal/cpp_tools/timers/include/ -std=c++17 $value 2> tmp/fix.out | python /home/p13ro/dev/cpp/github/include-what-you-use/fix_includes.py --nocomments --basedir=/home/p13ro/dev/cpp/gitlab/ScalFMM/ < tmp/fix.out
done
