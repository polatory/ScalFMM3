// ----------------------------------------------------------------
// File: bench/fmm-computation-seq.cpp
// ----------------------------------------------------------------

#include "scalfmm/options/options.hpp"

static auto fmm_options = scalfmm::options::_s(scalfmm::options::seq_timit);

#include "fmm-computation.hpp"
