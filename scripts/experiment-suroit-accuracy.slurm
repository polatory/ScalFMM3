#!/usr/bin/env bash
#SBATCH --job-name=scalfmm-suroit-accuracy
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -C "suroit"
#SBATCH --output=scalfmm-suroit-accuracy.log

manifest_file=".guix/scalfmm-manifest-gcc11-openblas.scm"
channels_file=".guix/scalfmm-channels.scm"

experiment_file="scripts/test-accuracy.sh"
guix time-machine -C "${channels_file}" -- shell --pure -m "${manifest_file}" -- bash "${experiment_file}" "${SLURM_JOB_NAME}"
