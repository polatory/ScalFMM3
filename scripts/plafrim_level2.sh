#!/usr/bin/env bash
set -x

exec guix time-machine -C .guix/scalfmm-channels.scm -- shell --pure --preserve="^CI|proxy$|^SLURM" \
  -m .guix/scalfmm-manifest-gcc-mkl-bench.scm \
  -- /bin/bash --norc ./scripts/plafrim_level3.sh
err=$?

# exit with error code from the guix command
exit $err
