// --------------------------------
// See LICENCE file at project root
// File : operators/m2m.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_M2M_HPP
#define SCALFMM_OPERATORS_M2M_HPP

#include "scalfmm/operators/interpolation/m2m_l2l.hpp"

namespace scalfmm::operators
{
    template<typename Approximation, typename Cell>
    inline void m2m(Approximation const& approximation, Cell const& child_cell, std::size_t child_index, Cell& parent_cell,
                    std::size_t tree_level = 2)
    {
        apply_m2m(approximation, child_cell, child_index, parent_cell, tree_level);
    }
}   // namespace scalfmm::operators
#endif   // SCALFMM_OPERATORS_M2M_HPP
