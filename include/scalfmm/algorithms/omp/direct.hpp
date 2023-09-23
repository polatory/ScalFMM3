// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/direct.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_DIRECT_HPP
#define SCALFMM_ALGORITHMS_OMP_DIRECT_HPP

#ifdef _OPENMP

#include <tuple>

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/omp/macro.hpp"
#include "scalfmm/algorithms/omp/priorities.hpp"
#include "scalfmm/lists/utils.hpp"
#include "scalfmm/operators/mutual_apply.hpp"
#include "scalfmm/operators/p2p.hpp"
#include "scalfmm/operators/tags.hpp"

namespace scalfmm::algorithms::omp::pass
{

    /// @brief compute the near field interaction when source=target.
    ///
    /// Compute the near field interaction when source=target, the p2p interactions lists
    /// should be construct before this call
    ///
    /// @tparam Tree
    /// @tparam NearField
    /// @param tree  the octree
    /// @param nearfield the nearfield operator
    ///
    /**
 * @brief Compute direct interaction between particles when the source tree and the target tree are the same
 *

 * @param tree the source/target tree
 * @param nearfield  the near-field operator
 */
    template<typename Tree, typename NearField>
    inline auto direct(Tree const& tree, NearField const& nearfield) -> void
    {
        using operators::p2p_full_mutual;
        using operators::p2p_inner;
        using operators::p2p_outer;

        // static constexpr int limit = (Tree::dimension - 1) * 6 + (Tree::dimension - 1);
        const auto separation_criterion = nearfield.separation_criterion();
        const auto mutual = nearfield.mutual();
        const auto period = tree.box().get_periodicity();
        const auto box_width = tree.box_width();

        // move test to execute
        // p2p stage
        auto begin = std::begin(tree);
        auto end = std::end(tree);

        const auto& matrix_kernel = nearfield.matrix_kernel();
        // loop on the groups
        auto begin_groups{std::get<0>(begin)};
        const auto end_groups{std::get<0>(end)};
        const auto prio_big{priorities::p2p_big};
        const auto prio_small{priorities::p2p_small};
        if(mutual)
        {
            // First perform out_of_group interaction in mutual kernel
            while(begin_groups != end_groups)
            {
                std::size_t current_out_interaction{0};
                std::size_t first_out_interaction{0};
                std::size_t last_out_interaction{0};
                // auto const& outside_interactions = (*begin_groups)->csymbolics().outside_interactions;
                auto const& group_dependencies = (*begin_groups)->csymbolics().group_dependencies;
                const auto current_group_ptr_particles = (*begin_groups).get()->depends_update();
                //
                // Loop on the pointer of other groups involved in the interactions with the current group
                auto const& sym_g = (*begin_groups)->csymbolics();
                ///
                for(auto& other_group_ptr_part: group_dependencies)
                {
                    std::tie(first_out_interaction, last_out_interaction) = list::get_outside_interaction_range(
                      sym_g, other_group_ptr_part->csymbolics(), current_out_interaction);

                const auto other_group_ptr_particles = other_group_ptr_part->depends_update();
#pragma omp task untied default(none) shared(period, box_width, matrix_kernel) firstprivate(                           \
  begin_groups, current_group_ptr_particles, other_group_ptr_particles, first_out_interaction, last_out_interaction)        \
  depend(inout                                                                                                 \
         : current_group_ptr_particles[0], other_group_ptr_particles[0]) priority(prio_small)
                    {
                        // Get for each previous group the indices range of interacting components and apply the
                        // operator between them.
                        operators::apply_out_of_group_p2p(**begin_groups, first_out_interaction, last_out_interaction,
                                                          matrix_kernel, period, box_width);
                    }
                    current_out_interaction = last_out_interaction;
                }
                ++begin_groups;
            }
        }
        //
        begin_groups = std::get<0>(begin);
        while(begin_groups != end_groups)
        {
            const auto current_group_ptr_particles = (*begin_groups).get()->depends_update();
            // clang does not manage to pass the separation_criterion correctly if it is not firstprivate
#pragma omp task untied default(none) shared(period, box_width, matrix_kernel)                                         \
  firstprivate(begin_groups, separation_criterion, mutual) depend(inout                                        \
                                                                  : current_group_ptr_particles[0]) priority(prio_big)
            {
                // loop on the leaves of the current group
                for(std::size_t leaf_index = 0; leaf_index < (*begin_groups)->size(); ++leaf_index)
                {
                    auto& leaf = (*begin_groups)->component(leaf_index);
                    // get the group type to retrieve internal type definitions
                    // auto mine_index = leaf.index();

                    // Get interation infos
                    auto const& leaf_symbolics = leaf.csymbolics();
                    auto const& interaction_iterators = leaf_symbolics.interaction_iterators;
                    // auto const& interaction_indexes = leaf_symbolics.interaction_indexes;
                    if(separation_criterion == 1)
                    {
                        if(leaf_symbolics.number_of_neighbors > 0)
                        {
                            if(mutual)
                            {
                                // Optimization for mutual interation
                                // Mutual with neighbors + inner of current component
                                p2p_full_mutual(matrix_kernel, leaf, interaction_iterators,
                                                leaf_symbolics.existing_neighbors_in_group, period, box_width);
                            }
                            else
                            {
                                p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                            }
                        }
                    }
                    p2p_inner(matrix_kernel, leaf);
                }   // end of for_each on leaves
            }       // end task
            ++begin_groups;
        }
    }

    /**
     * @brief Compute direct interaction between particles
     *
     *  When tree_source = tree_target we call the direct(tree_source, nearfield) function
     *
     * @param tree_source the source tree
     * @param tree_target  tke target tree where we compute the field
     * @param nearfield  the near-field operator
     */
    template<typename TreeSource, typename TreeTarget, typename NearField>
    inline auto direct(TreeSource& tree_source, TreeTarget& tree_target, NearField const& nearfield) -> void
    {
        bool source_target{false};
        if constexpr(std::is_same_v<TreeSource, TreeTarget>)
        {
            source_target = (&tree_source == &tree_target);
        }
        if(source_target)
        {
            if constexpr(std::is_same_v<TreeSource, TreeTarget>)
            {
                direct(tree_source, nearfield);
            }
        }
        else
        {
            // source != target
            using operators::p2p_outer;

            const auto period = tree_source.box().get_periodicity();
            const auto box_width = tree_source.box_width();
            // move test to execute
            // p2p stage


            const auto& matrix_kernel = nearfield.matrix_kernel();

            static constexpr auto prio_big{priorities::p2p_big};
            // iterators on target leaves
            auto begin_groups{tree_target.begin_mine_leaves()};
            auto end_groups{tree_target.end_mine_leaves()};

            while(begin_groups != end_groups)
            {
                //   depends on ptr on first particles of the grp
                const auto current_group_ptr_particles = (*begin_groups).get()->depends_update();

#pragma omp task untied default(none) shared(period, box_width, matrix_kernel)                   \
  firstprivate(begin_groups) depend(inout                                                         \
                                                 : current_group_ptr_particles[0]) priority(prio_big)
                {// mutexinoutset
                    // loop on the leaves of the current group
                    for(std::size_t leaf_index = 0; leaf_index < (*begin_groups)->size(); ++leaf_index)
                    {
                        auto& leaf = (*begin_groups)->component(leaf_index);
                        // get the group type to retrieve internal type definitions

                        // Get interation infos
                        auto const& leaf_symbolics = leaf.csymbolics();
                        auto const& interaction_iterators = leaf_symbolics.interaction_iterators;
                        //
                        // No test on separation_criterion
                        if(leaf_symbolics.number_of_neighbors > 0)
                        {
                            p2p_outer(matrix_kernel, leaf, interaction_iterators, period, box_width);
                        }
                    }
                };
                ++begin_groups;
            }
        }   // end else
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_DIRECT_HPP
