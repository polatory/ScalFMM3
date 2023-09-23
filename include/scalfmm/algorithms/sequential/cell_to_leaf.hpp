// --------------------------------
// See LICENCE file at project root
// File : algorithm/sequential/cell_to_leaf.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_CELL_TO_LEAF_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_CELL_TO_LEAF_HPP

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/operators/l2p.hpp"
#include "scalfmm/utils/massert.hpp"

namespace scalfmm::algorithms::sequential::pass
{
    /// @brief This function applies the l2p operator from the lower cell level to the leaf level.
    ///
    /// It applies the far field on the particles.
    /// We pass here the entire FmmOperator, this allows
    /// us to catch combination of matrix kernels and trigger
    /// optimizations.
    ///
    /// @tparam Tree the tree type
    /// @tparam FmmOperator the FmmOperator type
    /// @param tree reference to the tree
    /// @param FmmOperator const reference to the fmm_operator
    ///
    template<typename Tree, typename FmmOperator>
    inline auto cell_to_leaf(Tree& tree, FmmOperator const& fmm_operator) -> void
    {
        using operators::l2p;
        auto begin = std::begin(tree);
        auto end = std::end(tree);
        auto group_of_leaf_begin = std::get<0>(begin);
        auto group_of_leaf_end = std::get<0>(end);
        auto tree_height = tree.height();
        auto& cells_at_leaf_level = *(std::get<1>(begin) + (tree_height - 1));
        auto group_of_cell_begin = std::begin(cells_at_leaf_level);
        auto group_of_cell_end = std::end(cells_at_leaf_level);

        auto const& far_field = fmm_operator.far_field();

        assertm((*group_of_leaf_begin)->size() == (*group_of_cell_begin)->size(),
                "cell_to_leaf : nb group of leaves and nb first level of group cells differs !");

        while(group_of_leaf_begin != group_of_leaf_end && group_of_cell_begin != group_of_cell_end)
        {
            {   // Can be a task(in:iterCells, inout:iterParticles)
                for(std::size_t leaf_index = 0; leaf_index < (*group_of_leaf_begin)->size(); ++leaf_index)
                {
                    auto const& source_cell = (*group_of_cell_begin)->ccomponent(leaf_index);
                    auto& target_leaf = (*group_of_leaf_begin)->component(leaf_index);
                    l2p(far_field /*fmm_operator*/, source_cell, target_leaf);
                }
            }

            ++group_of_leaf_begin;
            ++group_of_cell_begin;
        }
    }
}   // namespace scalfmm::algorithms::sequential::pass

#endif
