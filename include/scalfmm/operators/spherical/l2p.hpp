// --------------------------------
// See LICENCE file at project root
// File : operators/spherical/l2p.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_SPHERICAL_L2P_HPP
#define SCALFMM_OPERATORS_SPHERICAL_L2P_HPP

#include "scalfmm/spherical/sperical.hpp"

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // l2p operator
    // -------------------------------------------------------------
    template<typename T, typename Cell, typename Leaf>
    inline auto apply_l2p(spherical::spherical<T> const& far_field, Cell const& source_cell, Leaf& target_leaf) -> void
    {
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_INTERPOLATION_L2P_HPP
