// --------------------------------
// See LICENCE file at project root
// File : utils/massert.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_MASSERT_HPP
#define SCALFMM_UTILS_MASSERT_HPP

#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp));

#endif   // SCALFMM_UTILS_MASSERT_HPP
