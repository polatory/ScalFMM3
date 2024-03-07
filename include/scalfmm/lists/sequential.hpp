#ifndef SCALFMM_LISTS_SEQUENTIAL_HPP
#define SCALFMM_LISTS_SEQUENTIAL_HPP

#include <scalfmm/lists/utils.hpp>
#include <scalfmm/utils/io_helpers.hpp>

namespace scalfmm::list::sequential
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
        //
        if constexpr(std::is_same_v<std::decay_t<TREE_S>, std::decay_t<TREE_T>>)
        {
            source_target = (&tree_source == &tree_target);
        }
        if((!source_target) && mutual)
        {
            throw std::invalid_argument(
              " Mutual set to true is prohibited when the sources are different from the targets.\n");
        }
        //std::cout << std::boolalpha << "source == target " << source_target << std::endl;
        const auto& period = tree_target.box().get_periodicity();
        const auto leaf_level = tree_target.leaf_level();
        auto begin_of_source_groups = std::get<0>(tree_source.begin());
        auto end_of_source_groups = std::get<0>(tree_source.end());

        // Iterate on the group of leaves I own
        component::for_each(
          // std::get<0>(tree_target.begin()), std::get<0>(tree_target.end()),
          tree_target.begin_mine_leaves(), tree_target.end_mine_leaves(),
          [&period, &neighbour_separation, &begin_of_source_groups, &end_of_source_groups, &leaf_level, mutual,
           source_target](auto& group_target)
          {
              std::size_t index_in_group{0};
              // Iterate on  leaves inside the group
              component::for_each(std::begin(*group_target), std::end(*group_target),
                                  [&group_target, &index_in_group, &begin_of_source_groups, &end_of_source_groups,
                                   &period, &neighbour_separation, &leaf_level, mutual, source_target](auto& leaf)
                                  {
                                      scalfmm::list::build_p2p_interaction_list_inside_group(
                                        leaf, begin_of_source_groups, end_of_source_groups, *group_target,
                                        index_in_group, leaf_level, mutual, period, neighbour_separation,
                                        source_target);

                                      ++index_in_group;
                                  });
              //
              //  if constexpr(!std::is_same_v<TREE_S, TREE_T>)
              {
                  if(mutual)
                  {
                      // here we are in the case that source == target and mutual interactions (optimization)
                      scalfmm::list::build_out_of_group_interactions(begin_of_source_groups, *group_target);
                  }
              }
          });

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
    inline auto build_m2l_interaction_list(TREE_S& tree_source, TREE_T& tree_target, const int& neighbour_separation) -> void
    {
        // Iterate on the group of leaves
        // here we get the first level of cells (leaf_level up to the top_level)
        auto tree_height = tree_target.height();
        int leaf_level = int(tree_height) - 1;
        auto const& period = tree_target.box().get_periodicity();
        auto const top_level = tree_target.box().is_periodic() ? 1 : 2;
        //
        for(int level = leaf_level; level >= top_level; --level)
        {
            // target
            auto group_of_cell_begin = tree_target.begin_mine_cells(level);
            auto group_of_cell_end = tree_target.end_mine_cells(level);
            // source
            auto begin_of_source_cell_groups = tree_source.begin_cells(level);
            auto end_of_source_cell_groups = tree_source.end_cells(level);
            // loop on target group
            component::for_each(
              group_of_cell_begin, group_of_cell_end,
              [begin_of_source_cell_groups, end_of_source_cell_groups, level, &period,
               &neighbour_separation](auto& group_target)
              {
                  // loop on target cell group
                  component::for_each(std::begin(*group_target), std::end(*group_target),
                                      [&group_target, begin_of_source_cell_groups, end_of_source_cell_groups, level,
                                       &period, &neighbour_separation](auto& cell_target)
                                      {
                                          list::build_m2l_interaction_list_for_group(
                                            *group_target, cell_target, begin_of_source_cell_groups,
                                            end_of_source_cell_groups, level, period, neighbour_separation);
                                      });
              });
        }
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
        std::cout << "start sequential::build_interaction_lists \n";

        build_p2p_interaction_list(tree_source, tree_target, neighbour_separation, mutual);
        std::cout << "  build_m2l_interaction_list\n";

        build_m2l_interaction_list(tree_source, tree_target, neighbour_separation);
        std::cout << "end sequential::build_interaction_lists \n";
    }

}   // namespace scalfmm::list::sequential
#endif   // SCALFMM_LISTS_SEQUENTIAL_HPP
