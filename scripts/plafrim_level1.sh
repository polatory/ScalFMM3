#!/usr/bin/env bash
set -x

# Submit slurm jobs

# number of machine nodes
SLURM_NNODES=1
# number of MPI tasks
SLURM_NTASKS=1
# number of CPU cores per task
SLURM_NCORES=1
SLURM_TIME=02:00:00

# to change defaults slurm parameters if needed
case "${CI_BENCHMARK}" in
  accuracy)
    SLURM_NAME=scalfmm_accuracy
    ;;
  timeseq)
    SLURM_NAME=scalfmm_timeseq
    ;;
  timeomp)
    SLURM_NAME=scalfmm_timeomp
    SLURM_NCORES=32
    ;;
  *)
    echo "CI_BENCHMARK is set to an unknown value."
    exit 1
    ;;
esac

sbatch --wait --job-name=${SLURM_NAME} --output=${SLURM_NAME}.log -N ${SLURM_NNODES} -n ${SLURM_NTASKS} -c ${SLURM_NCORES} \
       --time=${SLURM_TIME} --constraint=${CI_HOSTNAME} ./scripts/plafrim_level2.sh
err=$?

cat scalfmm.log

# exit with error code from the sbatch command
exit $err
