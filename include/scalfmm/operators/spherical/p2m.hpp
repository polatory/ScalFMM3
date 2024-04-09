// --------------------------------
// See LICENCE file at project root
// File : operators/spherical/p2m.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_SPHERICAL_P2M_HPP
#define SCALFMM_OPERATORS_SPHERICAL_P2M_HPP

#include "scalfmm/spherical/sperical.hpp"

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // P2M operator
    // -------------------------------------------------------------
    template<typename D, typename E, typename Leaf, typename Cell>
    inline auto apply_p2m(spherical::spherical<T> const& interp, Leaf const& source_leaf, Cell& cell) -> void
    {
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_SPHERICAL_P2M_HPP
