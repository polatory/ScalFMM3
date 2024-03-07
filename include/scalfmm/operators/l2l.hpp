// --------------------------------
// See LICENCE file at project root
// File : operators/l2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_L2L_HPP
#define SCALFMM_OPERATORS_L2L_HPP

#include "scalfmm/operators/interpolation/m2m_l2l.hpp"

namespace scalfmm::operators
{
    template<typename Approximation, typename Cell>
    inline void l2l(Approximation const& approximation, Cell const& parent_cell, std::size_t child_index, Cell& child_cell,
                    std::size_t tree_level = 2)
    {
        apply_l2l(approximation, parent_cell, child_index, child_cell, tree_level);
    }
}
#endif // SCALFMM_OPERATORS_M2M_HPP
