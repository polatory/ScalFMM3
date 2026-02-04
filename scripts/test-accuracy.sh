#!/usr/bin/env bash

export HFI_NO_CPUAFFINITY=1

suffix="$1"

# configuration and compilation
build_dir="build-${suffix}"
#rm -rf "${build_dir}"  # start from scratch
cmake -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" -Dscalfmm_BUILD_BENCH=ON
cmake --build "${build_dir}" --target fmm-computation-seq
exec_file="${build_dir}/bench/Release/fmm-computation-seq"

# changing parameters
dimension=(1 2 3)
order_values=(3 4 5 6 7 8 9 10 11 12)
tree_height=(3 4 5)
interp_settings_values=(0 1 2 3 4)
kernel_values=(0 6 7)

# fixed parameters
log_file="tmp-log-buffer.txt"
input_file="data/cuboid11-dim2-in-val1-center00-N10000.fma"
group_size=1
nb_threads=4  # for direct computation
nb_runs=1

result_folder="bench/results"
result_file_base="accuracy"
result_file_suffix="$(hostname)${suffix}"

mkdir -p "${result_folder}"

# main loop
for kernel in "${kernel_values[@]}";
do

    for interp_settings in "${interp_settings_values[@]}";
    do

	result_file="${result_folder}/${result_file_base}-interp${interp_settings}-kernel${kernel}-${result_file_suffix}.txt"

	# start from scratch
	rm -f "${result_file}"

	for order in "${order_values[@]}";
	do
	    echo -e "\t--- order = ${order} ---"

	    "${exec_file}" --dimension "${dimension}" \
			   --fmm-computation \
			   --direct-computation \
			   --order "${order}" \
			   --tree-height "${tree_height}" \
			   --group-size "${group_size}" \
			   --nb-runs "${nb_runs}" \
			   --kernel "${kernel}" \
			   --threads "${nb_threads}" \
			   --interp-settings "${interp_settings}" | tee "${log_file}"

	    python3 scripts/post-process-error.py --input-file "${log_file}" --result-file "${result_file}" -vv

	done

    done

done

rm -f "${log_file}"
