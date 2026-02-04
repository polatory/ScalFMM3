// ----------------------------------------------------------------
// File: bench/fmm-computation-omp.cpp
// ----------------------------------------------------------------

#include "scalfmm/options/options.hpp"

static auto fmm_options = scalfmm::options::_s(scalfmm::options::omp_timit);

#include "fmm-computation.hpp"
