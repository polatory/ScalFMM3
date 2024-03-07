#ifndef SCALFMM_LISTS_OMP_HPP
#define SCALFMM_LISTS_OMP_HPP

#define SCALFMM_LISTS_OMP
#include "scalfmm/algorithms/omp/priorities.hpp"
#include <scalfmm/lists/utils.hpp>
#include <scalfmm/utils/io_helpers.hpp>

namespace scalfmm::list::omp
{
    /**
     * @brief Construct the P2P  interaction list for the target tree
     *
     * @param[in] tree_source the tree containing the sources
     * @param[inout] tree_target the tree containing the targets.
     * @param[in] neighbour_separation separation criterion use to separate teh near and the far field
     * @param[in] mutual boolean to specify if the direct pass use a symmetric algorithm (mutual interactions)
     */
    template<typename TREE_S, typename TREE_T>
    inline auto build_p2p_interaction_list(TREE_S const& tree_source, TREE_T& tree_target,
                                           const int& neighbour_separation, const bool mutual) -> void
    {
        // We iterate on the leaves
        // true if source == target
        bool source_target{false};
        if constexpr(std::is_same_v<std::decay_t<TREE_S>, std::decay_t<TREE_T>>)
        {
            source_target = (&tree_source == &tree_target);
        }
        if((!source_target) && mutual)
        {
            throw std::invalid_argument(
              " Mutual set to true is prohibited when the sources are different from the targets.\n");
        }
        // std::cout << std::boolalpha << "source == target " << source_target << std::endl;
        const auto& period = tree_target.box().get_periodicity();
        const auto leaf_level = tree_target.leaf_level();
        auto begin_of_source_groups = std::get<0>(tree_source.begin());
        auto end_of_source_groups = std::get<0>(tree_source.end());

        // Iterate on the group of leaves I own
        component::for_each(
          tree_target.begin_mine_leaves(), tree_target.end_mine_leaves(),
          [&period, &neighbour_separation, &begin_of_source_groups, &end_of_source_groups, &leaf_level, mutual,
           source_target](auto& group_target)
          {
              auto group_target_ptr = group_target;
              static constexpr int prio{scalfmm::algorithms::omp::priorities::max};
#pragma omp task untied default(none) firstprivate(begin_of_source_groups, end_of_source_groups, group_target_ptr)     \
  shared(leaf_level, mutual, source_target, period, neighbour_separation) priority(prio)
              {
                  std::size_t index_in_group{0};

                  // Iterate on  leaves inside the group
                  component::for_each(
                    std::begin(*group_target_ptr), std::end(*group_target_ptr),
                    [&group_target_ptr, &index_in_group, &begin_of_source_groups, &end_of_source_groups, &period,
                     &neighbour_separation, &leaf_level, mutual, source_target](auto& leaf_target)
                    {
                        scalfmm::list::build_p2p_interaction_list_inside_group(
                          leaf_target, begin_of_source_groups, end_of_source_groups, *group_target_ptr, index_in_group,
                          leaf_level, mutual, period, neighbour_separation, source_target);

                        ++index_in_group;
                    });
                  //
                  if(mutual)
                  {
                      // here we are in the case that source == target and mutual interactions (optimization)
                      scalfmm::list::build_out_of_group_interactions(begin_of_source_groups, *group_target_ptr);
                  }
              }
          });
#pragma omp taskwait
        tree_target.is_interaction_p2p_lists_built() = true;
    }
    /**
     * @brief Construct the M2L interaction list for the target tree
     *
     * @param[in] tree_source the tree containing the sources
     * @param[inout] tree_target the tree containing the targets.
     * @param[in] neighbour_separation separation criterion use to separate teh near and the far field
     */
    template<typename TREE_S, typename TREE_T>
    inline auto build_m2l_interaction_list(TREE_S& tree_source, TREE_T& tree_target, const int& neighbour_separation)
      -> void
    {
        // Iterate on the group of leaves
        // here we get the first level of cells (leaf_level up to the top_level)
        auto tree_height = tree_target.height();
        int leaf_level = int(tree_height) - 1;
        auto const& period = tree_target.box().get_periodicity();
        auto const& top_level = tree_target.box().is_periodic() ? 1 : 2;
        //
        // auto cell_source_level_it = std::get<1>(tree_source.begin()) + leaf_level;
        for(int level = leaf_level; level >= top_level; --level)
        {
            // target
            auto group_of_cell_begin = tree_target.begin_mine_cells(level);
            auto group_of_cell_end = tree_target.end_mine_cells(level);
            // source
            // auto begin_of_source_cell_groups = std::begin(*cell_source_level_it);
            // auto end_of_source_cell_groups = std::end(*cell_source_level_it);
            auto begin_of_source_cell_groups = tree_source.begin_cells(level);
            auto end_of_source_cell_groups = tree_source.end_cells(level);
            // loop on target group

            component::for_each(group_of_cell_begin, group_of_cell_end,
                                [begin_of_source_cell_groups, end_of_source_cell_groups, level, &period,
                                 &neighbour_separation](auto& group_target)
                                {
                                    static constexpr int prio{scalfmm::algorithms::omp::priorities::max};

#pragma omp task untied default(none)                                                                                  \
  firstprivate(group_target, begin_of_source_cell_groups, end_of_source_cell_groups, level)                            \
    shared(period, neighbour_separation) priority(prio)
                                    {
                                        // loop on target cell group
                                        component::for_each(
                                          std::begin(*group_target), std::end(*group_target),
                                          [&group_target, begin_of_source_cell_groups, end_of_source_cell_groups, level,
                                           &period, &neighbour_separation](auto& cell)
                                          {
                                              list::build_m2l_interaction_list_for_group(
                                                *group_target, cell, begin_of_source_cell_groups,
                                                end_of_source_cell_groups, level, period, neighbour_separation);
                                          });
                                    }
                                });

            // --cell_source_level_it;
        }
#pragma omp taskwait
        tree_target.is_interaction_m2l_lists_built() = true;
    }

    /**
     * @brief Construct the P2P and the M2L interaction lists for the target tree
     *
     * @param[in] tree_source the tree containing the sources
     * @param[inout] tree_target the tree containing the targets.
     * @param[in] neighbour_separation separation criterion use to separate teh near and the far field
     * @param[in] mutual boolean to specify if the direct pass use a symmetric algorithm (mutual interactions)
     */
    template<typename TREE_S, typename TREE_T>
    inline auto build_interaction_lists(TREE_S& tree_source, TREE_T& tree_target, const int& neighbour_separation,
                                        const bool mutual) -> void
    {
#pragma omp parallel default(none) shared(tree_source, tree_target, neighbour_separation, mutual)
        {
#pragma omp single nowait
            {
                build_m2l_interaction_list(tree_source, tree_target, neighbour_separation);
            }
#pragma omp single
            {
                build_p2p_interaction_list(tree_source, tree_target, neighbour_separation, mutual);
            }
        }
    }

}   // namespace scalfmm::list::omp
#endif   // SCALFMM_LISTS_OMP_HPP
