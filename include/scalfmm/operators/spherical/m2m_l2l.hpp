// --------------------------------
// See LICENCE file at project root
// File : operators/spherical/m2m_l2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_SPHERICAL_M2M_L2L_HPP
#define SCALFMM_OPERATORS_SPHERICAL_M2M_L2L_HPP

#include "scalfmm/spherical/sperical.hpp"

namespace scalfmm::operators
{
    // -------------------------------------------------------------
    // m2m operator
    // -------------------------------------------------------------
    template<typename T, typename Cell>
    inline auto apply_m2m(spherical::spherical<T> const& far_field, Cell const& child_cell, Cell& parent_cell) -> void
    {
    }
    // -------------------------------------------------------------
    // l2l operator
    // -------------------------------------------------------------
    template<typename T, typename Cell>
    inline auto apply_l2l(spherical::spherical<T> const& far_field, Cell const& parent_cell, Cell& child_cell) -> void
    {
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_SPHERICAL_M2M_L2L_HPP
