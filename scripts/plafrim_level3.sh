#!/usr/bin/env bash
set -ex

export HFI_NO_CPUAFFINITY=1
export MKLROOT=$GUIX_ENVIRONMENT

if [[ -z "${CI_COMMIT_SHA}" ]]; then
  export GIT_COMMIT=`git rev-parse HEAD`
else
  export GIT_COMMIT=$CI_COMMIT_SHA
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0=safe.directory
  export GIT_CONFIG_VALUE_0=*
fi
export GIT_COMMIT_DATE=`git show --no-patch --format=%ci $GIT_COMMIT`

if [[ -z "${CI_HOSTNAME}" ]]; then
  export CI_HOSTNAME=`hostname`
fi

export SCALFMM_EXE_DIR=$PWD/build/bench/Release

# clean old benchmarks
if [ -d scripts/results ]; then
  rm scripts/results -r
fi

# configuration
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" -Dscalfmm_BUILD_BENCH=ON -Dscalfmm_USE_MKL=ON

# compilation and run benchmarks
case "${CI_BENCHMARK}" in
  accuracy)
    echo "CI_BENCHMARK is set to accuracy."
    # compilation
    cmake --build build --target fmm-computation-seq
    # run benchmarks
    jube run scripts/scalfmm-accuracy.xml --id 1
    # parse result log files
    jube analyse scripts/results --id 1
    # generate csv file
    jube result scripts/results --id 1 -o accuracy_csv > scalfmm_accuracy.csv
    cat scalfmm_accuracy.csv
    # update scalfmm.sqlite3 database, table accuracy
    #jube result scripts/results --id 1 -o accuracy
    ;;
  timeseq)
    echo "CI_BENCHMARK is set to timeseq."
    cmake --build build --target fmm-computation-seq
    jube run scripts/scalfmm-time-seq.xml --id 2
    jube analyse scripts/results --id 2
    jube result scripts/results --id 2 -o timeseq_csv > scalfmm_timeseq.csv
    cat scalfmm_timeseq.csv
    #jube result scripts/results --id 2 -o timeseq
    ;;
  timeomp)
    echo "CI_BENCHMARK is set to timeomp."
    cmake --build build --target fmm-computation-omp
    jube run scripts/scalfmm-time-omp.xml --id 3
    jube analyse scripts/results --id 3
    jube result scripts/results --id 3 -o timeomp_csv > scalfmm_timeomp.csv
    cat scalfmm_timeomp.csv
    #jube result scripts/results --id 3 -o timeomp
    ;;
  *)
    echo "CI_BENCHMARK is set to an unknown value."
    exit 1
    ;;
esac
