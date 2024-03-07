// --------------------------------
// See LICENCE file at project root
// File : operators/p2m.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_P2M_HPP
#define SCALFMM_OPERATORS_P2M_HPP

#include "scalfmm/operators/interpolation/p2m.hpp"

namespace scalfmm::operators
{
    /**
     * @brief This function interpolates the inputs of the particle on the interpolation grid.
     *
     * @param far_field   the far-field operator
     * @param leaf  the source leaf
     * @param cell  the source cell
     */
    template<typename FarField, typename Leaf, typename Cell>
    inline void p2m(FarField const& far_field, Leaf const& leaf, Cell& cell)
    {
        apply_p2m(far_field, leaf, cell);
    }
}   // namespace scalfmm::operators
#endif   // SCALFMM_OPERATORS_P2M_HPP
