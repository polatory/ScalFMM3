#!/usr/bin/env bash

export HFI_NO_CPUAFFINITY=1

suffix="$1"

# configuration and compilation
build_dir="build-${suffix}"
#rm -rf "${build_dir}"  # start from scratch
cmake -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" -Dscalfmm_BUILD_BENCH=ON
cmake --build "${build_dir}" --target fmm-computation-omp
exec_file="${build_dir}/bench/Release/fmm-computation-omp"

# changing parameters
dimension=(1 2 3)
nb_threads_values=(1 2 4 8 16)
group_size_values=(1 5 10 100 1000)
tree_height_values=(3 4 5)

# fixed parameters
log_file="tmp-log-buffer.txt"
kernel=0
order=5
nb_runs=5
interp_settings=3
input_file="data/cuboid11-dim2-in-val1-center00-N10000.fma"

# set OpenMP environment variables
export OMP_PROC_BIND=true
export OMP_MAX_TASK_PRIORITY=11

result_folder="bench/results"
result_file_base="time-task-dep"
result_file_suffix="$(hostname)${suffix}"

mkdir -p "${result_folder}"

# main loop
for group_size in "${group_size_values[@]}";
do

    # start from scratch
    rm -f "${result_file}"

    result_file="${result_folder}/${result_file_base}-interp${interp_settings}-kernel${kernel}-group-size${group_size}-${result_file_suffix}.txt"

    for nb_threads in "${nb_threads_values[@]}";
    do

	for tree_height in "${tree_height_values[@]}";
	do

	    "${exec_file}" --dimension "${dimension}" \
			   --fmm-computation \
			   --order "${order}" \
			   --tree-height "${tree_height}" \
			   --group-size "${group_size}" \
			   --nb-runs "${nb_runs}" \
			   --kernel "${kernel}" \
			   --threads "${nb_threads}" \
			   --interp-settings "${interp_settings}" | tee "${log_file}"

	    python3 scripts/post-process-omp.py --input-file "${log_file}" --result-file "${result_file}" -vv

	done

    done

done

rm -f "${log_file}"
