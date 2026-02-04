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
order_values=(5)
interp_settings_values=(1 2 4)
size_values=(1000 5000 10000 50000 100000 500000 1000000)
tree_height_values=(3 4 5)
kernel_values=(0)

# fixed parameters
log_file="tmp-log-buffer.txt"
nb_threads=1  # fully sequential !
group_size=1
nb_runs=5

result_folder="bench/results"
result_file_base="time-seq"
result_file_suffix="$(hostname)${suffix}"

mkdir -p "${result_folder}"

# main loop
for kernel in "${kernel_values[@]}";
do

    for interp_settings in "${interp_settings_values[@]}";
    do

	# start from scratch
	rm -f "${result_file}"

        result_file="${result_folder}/${result_file_base}-interp${interp_settings}-kernel${kernel}-${result_file_suffix}.txt"

	for size in "${size_values[@]}";
	do

	    for tree_height in "${tree_height_values[@]}";
	    do

		for order in "${order_values[@]}";
		do
		    echo -e "\t--- order = ${order} ---"

		    "${exec_file}" --dimension "${dimension}" \
				   --fmm-computation \
				   --order "${order}" \
				   --tree-height "${tree_height}" \
				   --size "${size}" \
				   --group-size "${group_size}" \
				   --nb-runs "${nb_runs}" \
				   --kernel "${kernel}" \
				   --threads "${nb_threads}" \
				   --interp-settings "${interp_settings}" | tee "${log_file}"

		    python3 scripts/post-process-seq.py --input-file "${log_file}" --result-file "${result_file}" -vv

		done

	    done

	done

    done

done

rm -f "${log_file}"
