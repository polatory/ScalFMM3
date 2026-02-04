#!/usr/bin/env bash

# setup
exec_file="build-default/bench/Release/fmm-computation-seq"

# changing parameters
result_folder="bench/results"
result_file_base="accuracy"
order_values=(3 4 5 6 7 8 9 10)
interp_settings_values=(0 1 2 3 4)
kernel_values=(0 6 7)

# fixed parameters
log_file="tmp-log-buffer.txt"
input_file="data/cuboid11-dim2-in-val1-center00-N10000.fma"
dimension=2
tree_height=4
group_size=1
nb_runs=1

mkdir -p "${result_folder}"

# main loop
for kernel in "${kernel_values[@]}";
do

    for interp_settings in "${interp_settings_values[@]}";
    do

	result_file="${result_folder}/${result_file_base}-interp${interp_settings}-kernel${kernel}.txt"

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
			   --input-file "${input_file}" \
			   --interp-settings "${interp_settings}" | tee "${log_file}"

	    python3 bench/post-process-error.py --input-file "${log_file}" --result-file "${result_file}" -vv

	done

    done

done
