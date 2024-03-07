// --------------------------------
// See LICENCE file at project root
// File : algorithm/sequential/direct.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_DIRECT_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_DIRECT_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/operators/l2p.hpp"
#include "scalfmm/operators/mutual_apply.hpp"
#include "scalfmm/operators/p2p.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

namespace scalfmm::algorithms::sequential::pass
{
    /// @brief Applies the near-field on the leaves of the tree.
    /// The pass will update the leaves' interaction lists, retrieve the
    /// neighbors for each leaf and apply the p2p operator on the list.
    ///
    /// @tparam TreeSource source tree type
    /// @tparam TreeTarget target tree type
    /// @tparam NearField near_field type
    /// @param tree_source a reference to the tree containing the sources
    /// @param tree_target a reference to the tree containing the target
    /// @param nearfield a const reference of the near-field
    ///
    /// @return void

    template<typename TreeSource, typename TreeTarget, typename NearField>
    inline auto direct(TreeSource& tree_source, TreeTarget& tree_target, NearField const& nearfield) -> void
    {
        bool source_target{false};
        if constexpr(std::is_same_v<TreeSource, TreeTarget>)
        {
            source_target = (&tree_source == &tree_target);
        }
        {
            using operators::p2p_full_mutual;
            using operators::p2p_inner;
            using operators::p2p_outer;

            // static constexpr int limit = (Tree::dimension - 1) * 6 + (Tree::dimension - 1);
            const auto separation_criterion = nearfield.separation_criterion();
            // get parameter to check if we are using mutual interactions
            const auto mutual = nearfield.mutual();
            const auto period = tree_source.box().get_periodicity();
            const auto box_width = tree_source.box_width();

            // move test to execute
            // p2p stage
            const auto level_leaves = tree_target.height() - 1;
            // iterators on target leaves
            auto begin = std::begin(tree_target);
            auto end = std::end(tree_target);

            std::size_t number_of_groups_processed{0};
            const auto& matrix_kernel = nearfield.matrix_kernel();

            auto group_of_source_leaf_begin = std::get<0>(std::begin(tree_source));
            auto group_of_source_leaf_end = std::get<0>(std::end(tree_source));
            // loop on the groups
            auto l_group_p2p = [&matrix_kernel, begin, &level_leaves, group_of_source_leaf_begin,
                                group_of_source_leaf_end, &period, &box_width, &number_of_groups_processed,
                                separation_criterion, source_target, mutual](auto& group)
            {
                std::size_t index_in_group{0};
                // loop on the leaves of the current group
                auto l_leaf_p2p = [&matrix_kernel, &group, &index_in_group, &level_leaves, group_of_source_leaf_begin,
                                   group_of_source_leaf_end, &period, &box_width, separation_criterion, source_target,
                                   mutual](auto& leaf)
                {
                    // Get interation infos
                    auto const& leaf_symbolics = leaf.csymbolics();
                    auto const& interaction_iterators = leaf_symbolics.interaction_iterators;
                    //
                    // No test on separation_criterion
                    if(leaf_symbolics.number_of_neighbors > 0)
                    {
                        if constexpr(std::is_same_v<TreeSource, TreeTarget>)
                        {
                            // Here we need the same type.
                            if(source_target)   // sources == targets
                            {
                                // if(separation_criterion == 1)
                                {
                                    if(mutual)
                                    {
                                        // Optimization for mutual interactions
                                        // The p2p interaction list contains only indexes bellow mine.
                                        // Mutual with neighbors + inner of current component
                                        p2p_full_mutual(matrix_kernel, leaf, interaction_iterators,
                                                        leaf_symbolics.existing_neighbors_in_group, period, box_width);
                                    }
                                    else   // non mutual
                                    {
                                        p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                                        // Wrong call p2p_inner - the function uses mutual approach.
                                    }
                                    p2p_inner(matrix_kernel, leaf, mutual);
                                }
                            }
                            else   // sources != targets
                            {
                                p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                            }
                        }
                        else
                        {
                            p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                        }
                    }
                    else   // no neighbors, only interaction with the cell
                    {
                        if(source_target)   // sources == targets
                        {
                            p2p_inner(matrix_kernel, leaf, mutual);
                        }
                    }
                };
                // Evaluate inside the group
                component::for_each(std::begin(*group), std::end(*group), l_leaf_p2p);
                // if  the sources and the target are identical and we are using the mutual
                //  algorithm then we have to proceed teh interactions with leaves inside groups
                // bellow mine
                if constexpr(std::is_same_v<TreeSource, TreeTarget>)
                {
                    if(mutual && source_target)
                    {
                        operators::apply_out_of_group_p2p(*group, matrix_kernel, period, box_width);
                    }
                }
            };
            //
            // Evaluate  P2P for all group group
            component::for_each(std::get<0>(begin), std::get<0>(end), l_group_p2p);
        }
    }
}   // namespace scalfmm::algorithms::sequential::pass

#endif   // SCALFMM_ALGORITHMS_SEQUENTIAL_DIRECT_HPP
