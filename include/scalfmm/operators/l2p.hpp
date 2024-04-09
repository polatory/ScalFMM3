// --------------------------------
// See LICENCE file at project root
// File : operators/l2p.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_L2P_HPP
#define SCALFMM_OPERATORS_L2P_HPP

#include "scalfmm/operators/interpolation/l2p.hpp"

namespace scalfmm::operators
{
    template<typename FarField, typename Cell, typename Leaf>
    inline void l2p(FarField const& far_field, Cell const& source_cell, Leaf& target_leaf)
    {
        apply_l2p(far_field, source_cell, target_leaf);
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_L2P_HPP
