// --------------------------------
// See LICENCE file at project root
// File : operators/m2l.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_M2L_HPP
#define SCALFMM_OPERATORS_M2L_HPP

#include <cstddef>

namespace scalfmm::operators
{
    /**
     * @brief Compute M2L operator between two cells (only used in periodic case)
     *
     * @tparam Approximation
     * @tparam Cell
     * @param approximation
     * @param source_cell
     * @param neighbor_idx
     * @param target_cell
     * @param current_tree_level
     * @param buffer
     */
    template<typename Approximation, typename Cell>
    inline void m2l(Approximation const& approximation, Cell const& source_cell, std::size_t neighbor_idx,
                    Cell& target_cell, std::size_t current_tree_level,
                    [[maybe_unused]] typename Approximation::buffer_type& buffer)
    {
        approximation.apply_m2l_single(source_cell, target_cell, neighbor_idx, current_tree_level, buffer);
    }

    template<typename Approximation, typename Cell>
    inline void m2l_loop(Approximation const& approximation, Cell& target_cell, std::size_t current_tree_level,
                         [[maybe_unused]] typename Approximation::buffer_type& buffer)
    {
        approximation.apply_m2l_loop(target_cell, current_tree_level, buffer);
    }
}   // namespace scalfmm::operators

#endif   // SCALFMM_OPERATORS_M2L_HPP
