// --------------------------------
// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/interaction_list.hpp
// --------------------------------
#ifndef SCALFMM_LISTS_UTIL_HPP
#define SCALFMM_LISTS_UTIL_HPP

#include <algorithm>
#include <iostream>
#include <tuple>

#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"

#include <cpp_tools/colors/colorized.hpp>

namespace scalfmm::list
{
    /**
     * @brief find the iterator of the component in (begin,end) with index idx
     *
     * @param begin start iterator on the groups
     * @param end end iterator on the groups
     * @param idx index to find
     * @return tuple containing  {position in the groups, iterator}
     */
    template<typename group_iterator_type, typename Index_type>
    inline auto find_component_group_near(group_iterator_type const& begin, group_iterator_type const& end,
                                          Index_type idx)
    {
        std::int64_t id_group{0};
        group_iterator_type found_group_it{};
        std::size_t found_group_id{0};
        auto b_leaf_it = begin;
        auto e_leaf_it = end;
        auto last_group = e_leaf_it - 1;
        if((*b_leaf_it)->symbolics().ending_index > idx)
        {
            return std::make_tuple(found_group_id, *b_leaf_it);
        }
        else if(idx >= (*last_group)->symbolics().ending_index)
        {
            found_group_id = std::distance(b_leaf_it, e_leaf_it) - 1;
            return std::make_tuple(found_group_id, *last_group);

        }
        // Iterate on the group of leaves
        for(group_iterator_type it = b_leaf_it; it != e_leaf_it; ++it)
        {
            if((*it)->is_below(idx))
            {
                found_group_id = id_group;
                found_group_it = it;
                break;
            }
            ++id_group;
        }
        return std::make_tuple(found_group_id, *found_group_it);
    }

    /**
     * @brief
     *
     * @param begin iterator on the first source group
     * @param end iterator on the last source group
     * @param idx current Morton index of the target components (cell/leaf)
     * @param group_source current source group
     * @param group_source_index curent index in the sources groups of the current groups
     * @param source_eq_target false if source and target particles are different ; true otherwise
     * @return a tuple containing tuple containing  {position in the groups, iterator}
     */
    template<typename group_iterator_type, typename Index_type, typename group_iterator1_type>
    inline auto find_component_group_near(group_iterator_type const& begin, group_iterator_type const& end,
                                          Index_type idx, group_iterator1_type group_source,
                                          std::size_t& group_source_index, bool source_eq_target)
    {
        if(source_eq_target)
        {
            return std::make_tuple(group_source_index, group_source);
        }
        return list::find_component_group_near(begin, end, idx);
    }

    //////////////////////////////////////////////////////////////////////////////////
    ///               Routines to build the M2L interaction list
    ///
    /// @ingroup update_interaction_list
    /// @brief This function takes a target cell and its level in the tree
    /// and updates the component's symbolics information : the morton indices
    /// of its neighbors and the number of neighbors. It returns the theoretical
    /// interaction list of the leaf.
    ///
    /// @tparam Component
    /// @param[in] t tag to select the good algorithm m2l or p2p
    /// @param[inout] component (target) cell
    /// @param[in] tree_level level in the tree to construct the interation list
    /// @param[in] period array of periodicity in each direction
    /// @param[in] neighbour_separation distance between neighbors and component
    ///
    template<typename Component, typename Array>
    inline auto build_symbolic_interaction_list(operators::impl::tag_m2l t, Component& component,
                                                std::size_t tree_level, Array const& period,
                                                const int& neighbour_separation) -> void
    {
        // Get symbolics infos on the group and the component
        auto& component_symbolics = component.symbolics();
        // Calculate the coordinate of the component from the morton index
        auto coordinate{
          index::get_coordinate_from_morton_index<Component::dimension>(component_symbolics.morton_index)};
        // update component's symbolics information
        std::tie(component_symbolics.interaction_indexes, component_symbolics.interaction_positions,
                 component_symbolics.number_of_neighbors) =
          index::get_m2l_list(coordinate, tree_level, period, neighbour_separation);
    }

    /**
     * @brief  Get iterator inside the group
     *
     * This function updates the interaction list in the target component symbolics with the source components available
     *  in the current group

     * @param[in] group the source group to find the index
     * @param[inout] target the target cell/leaf
     * @param[in] morton_index_interaction the morton index we search in group
     * @param[in] counter_existing_component
     * @param[in] interaction_index the index in the list of interaction
     *
     * @return true if iterator (morton index) found in the group
     * @return false otherwise
     */
    template<typename Group, typename Component>
    inline auto get_interacting_component_in(Group& group, Component& target,
                                             std::size_t morton_index_interaction,
                                             std::size_t counter_existing_component, std::size_t interaction_index)
      -> bool
    {
        // Get symbolics infos on the group and the component
        auto& component_symbolics = target.symbolics();
        // Get the position in group and check if the component exists
        const int leaf_position = group.component_index(morton_index_interaction);
        if(leaf_position != -1)
        {
            component_symbolics.set(counter_existing_component, interaction_index,
                                    group.component_iterator(leaf_position));
        }

        return (leaf_position != -1) ;
    }

    // This function updates the interaction list in the target component symbolics with the source components out of
    // the current group being proceed.
    template<typename GroupIterator, typename Component>
    inline auto get_interacting_component_out_left(GroupIterator group_begin, GroupIterator group_end,
                                                   Component& target, std::size_t morton_index_interaction,
                                                   std::size_t group_index, std::size_t counter_existing_component,
                                                   std::size_t index_interaction, std::size_t last_component_index)
    {
        // Get symbolics infos on the group and the component
        auto& component_symbolics = target.symbolics();
        // Iterators for going left and right
        auto left_group_it{group_begin};
        std::advance(left_group_it, group_index);

        int going_left{int(group_index)};
        bool found_interaction{false};
        int leaf_position{};

        const auto leaf_does_not_exist{-1};

        // Loop while you don't reach the begin and the end of the level
        while(going_left >= 0)
        {
            auto const& group_symbolics_l = (*left_group_it)->csymbolics();
            // Check if the morton index of the source is in this group
            if(group_symbolics_l.starting_index <= morton_index_interaction &&
               morton_index_interaction < group_symbolics_l.ending_index)
            {
                // Get the index of the source inside the group, if the morton index is not in the group return -1
                leaf_position = (*left_group_it)
                                  ->component_index(morton_index_interaction, last_component_index,
                                                    group_symbolics_l.number_of_component_in_group);
                if(leaf_position != leaf_does_not_exist)
                {
                    // Update list at the current counter of existing component.
                    component_symbolics.set(
                      counter_existing_component, index_interaction,
                      (*left_group_it)->component_iterator(static_cast<std::size_t>(leaf_position)));
                    found_interaction = true;
                    break;
                }
            }
            --left_group_it;
            --going_left;
        }

        // if not found you return the start of the previous search
        if(!found_interaction)
        {
            return std::make_tuple(found_interaction, group_index, last_component_index);
        }
        else
        {
            return std::make_tuple(found_interaction, static_cast<std::size_t>(going_left),
                                   static_cast<std::size_t>(leaf_position));
        }
    }
    // This function updates the interaction list in the target component symbolics with the source components out of
    // the current group being processed.
    /**
     * @brief Get the interacting component out right object
     *
     * @param group_begin
     * @param group_end
     * @param target
     * @param morton_index_interaction
     * @param group_index
     * @param counter_existing_component
     * @param index_interaction
     * @param last_component_index
     * @return auto
     */
    template<typename GroupIterator, typename Component>
    inline auto get_interacting_component_out_right(GroupIterator group_begin, GroupIterator group_end,
                                                    Component& target, std::size_t morton_index_interaction,
                                                    std::size_t group_index, std::size_t counter_existing_component,
                                                    std::size_t index_interaction, std::size_t last_component_index)
    {
        // Get symbolics infos on the group and the component
        auto& component_symbolics = target.symbolics();
        // Iterators for going left and right
        auto right_group_it{group_begin};
	long int number_of_groups{std::distance(group_begin, group_end)};
	std::size_t going_right{group_index};
        bool found_interaction{false};
        int leaf_position{};
        std::advance(right_group_it, going_right);

        const auto leaf_does_not_exist{-1};
        // Loop while you don't reach the begin and the end of the level
        while(going_right < std::size_t(number_of_groups))
        {
            auto const& group_symbolics_r = (*right_group_it)->csymbolics();
            if(group_symbolics_r.starting_index <= morton_index_interaction &&
               morton_index_interaction < group_symbolics_r.ending_index)
            {
                leaf_position = (*right_group_it)
                                  ->component_index(morton_index_interaction, last_component_index,
                                                    group_symbolics_r.number_of_component_in_group);
                if(leaf_position != leaf_does_not_exist)
                {
                    component_symbolics.set(counter_existing_component, index_interaction,
                                            (*right_group_it)->component_iterator(leaf_position));

                    found_interaction = true;
                    break;
                }
            }
            ++right_group_it;
            ++going_right;
            last_component_index = 0;
        }

        // if not found you return the start of the previous search
        if(!found_interaction)
        {
            return std::make_tuple(found_interaction, group_index, last_component_index);
        }
        else
        {
	  return std::make_tuple(found_interaction, static_cast<std::size_t>(going_right), static_cast<std::size_t>(leaf_position));
        }
    }
    /**
     * @brief Build group dependencies for a given target group
     *
     * @tparam GroupIterator
     * @tparam Storage
     * @param grp  current target group (containing the locals)
     * @param begin_of_groups iterator on the source group (containing the mutipoles)
     * @param group_index
     * @param last_group_index
     * @todo Merge begin_of_groups, std::size_t group_index ->  std::advance(begin_of_groups, last_group_index)
     *
     */
    template<typename GroupIterator, typename Storage>
    void build_m2l_dependencies(scalfmm::component::group<scalfmm::component::cell<Storage>>& grp,
                                GroupIterator begin_of_groups, std::size_t group_index,
                                std::size_t const& last_group_index)
    {
#ifdef _OPENMP
        auto found_in_group_it{begin_of_groups};
        std::advance(found_in_group_it, last_group_index);
        auto& g_d = grp.symbolics().group_dependencies_m2l;
        auto ptr = &(*found_in_group_it)->ccomponent(0).cmultipoles(0);
        // std ::cout << "    add depend " << ptr << std::endl;
        // io::print("       g_d ", g_d);
        //  if(last_group_index != group_index && std::end(g_d) == std::find(std::begin(g_d), std::end(g_d), ptr))
        if(std::end(g_d) == std::find(std::begin(g_d), std::end(g_d), ptr))
        {
            // std::cout << "        add ptr !\n";
            g_d.push_back(ptr);
            // io::print("g_d ", g_d);
        }
#endif
    }
    // general  version
    template<typename Group, typename GroupIterator>
    void build_m2l_dependencies(Group& group, GroupIterator begin_of_groups, std::size_t group_index,
                                std::size_t const& last_group_index)
    {
    }

    /**
     * @brief Updates the list of cell iterators in the M2L interaction list. Also updates dependencies for task-based
     * calculations
     *
     * @param begin_of_groups  begin iterator of the source groups
     * @param end_of_groups  end iterator of the source groups
     * @param group  group near or contains the Morton index of the component (target)
     * @param group_source_index  index of the group in the array on group
     * @param group_target  target containing the component (target)
     * @param cell_target  cell or leaf for which we construct the interaction list
     */
    template<typename GroupIterator, typename GroupT, typename GroupS, typename Component>
    inline auto build_interaction_list_iterators(GroupIterator begin_of_groups, GroupIterator end_of_groups,
                                                 GroupS& group, std::size_t group_source_index, GroupT& group_target,
                                                 Component& cell_target, bool m2l_list) -> void
    {
        // group = source group
        auto& group_symbolics = group.symbolics();
        // cell target
        auto& component_symbolics = cell_target.symbolics();
        auto& interaction_indexes = component_symbolics.interaction_indexes;

        std::size_t counter_existing_component{0};   // the counter position of the existing iterator
        std::size_t last_group_index{group_source_index};
        std::size_t last_component_index{0};
        bool first_out_left{true};
        int pos{0};
        // Loop on the number of neighbors
        for(size_t index_interaction = 0; index_interaction < component_symbolics.number_of_neighbors; ++index_interaction)
        {
            bool found{false};
            // morton index of the interaction
            std::size_t morton_index_interaction = interaction_indexes.at(static_cast<std::size_t>(index_interaction));
            // std::cout << "   search for source " << morton_index_interaction << std::endl;

            if(group_symbolics.starting_index <= morton_index_interaction &&
               morton_index_interaction < group_symbolics.ending_index)
            {
                // Get the morton index of the neighbor
                // std::cout << "     Check inside the group [" << group_symbolics.starting_index << ", "
                //           << group_symbolics.ending_index << "]\n";
                if(list::get_interacting_component_in(group, cell_target, morton_index_interaction,
                                                      counter_existing_component, index_interaction))
                {
                    last_group_index = group_symbolics.idx_global;
                    found = true;
                    // std::cout << "                found !\n";
                    // ++counter_existing_component;
                }
            }
            // If we are out of block
            else
            {
                // wo go on the left
                if(morton_index_interaction < group_symbolics.starting_index)
                {
                    // if we look the first interaction out of the group on the left we go down.
                    if(first_out_left)
                    {
                        std::tie(found, last_group_index, last_component_index) =
                          list::get_interacting_component_out_left(
                            begin_of_groups, end_of_groups, cell_target, morton_index_interaction, last_group_index,
                            counter_existing_component, index_interaction, last_component_index);
                        if(found)
                        {
                            first_out_left = false;
                            last_component_index = 0;
                            // ++counter_existing_component;
                        }
                    }
                    // Then we go up to find the others
                    else
                    {
                        std::tie(found, last_group_index, last_component_index) =
                          list::get_interacting_component_out_right(
                            begin_of_groups, end_of_groups, cell_target, morton_index_interaction, last_group_index,
                            counter_existing_component, index_interaction, last_component_index);
                        // if(found)
                        // {
                        //     ++counter_existing_component;
                        // }
                    }
                }
                // or we go on the right
                else if(morton_index_interaction >= group_symbolics.ending_index)
                {
                    std::tie(found, last_group_index, last_component_index) = list::get_interacting_component_out_right(
                      begin_of_groups, end_of_groups, cell_target, morton_index_interaction, last_group_index,
                      counter_existing_component, index_interaction, last_component_index);
                    // if(found)
                    // {
                    //     ++counter_existing_component;
                    // }
                }
            }
            if(found)
            {
                // std::cout << "   Found  in group source " << group_source_index << "  last_group_index "
                //           << last_group_index << "  last_component_index " << last_component_index << "  pos " << pos
                //           << std::endl;
                ++counter_existing_component;
                interaction_indexes.at(pos) = morton_index_interaction;
                ++pos;
#ifdef _OPENMP
                if(m2l_list)
                {
                    build_m2l_dependencies(group_target, begin_of_groups, group_source_index, last_group_index);
                }
#endif
            }
        }
        component_symbolics.finalize(true, counter_existing_component);
    }

    // /**
    //  * @ingroup update_interaction_list
    //  * @brief update the M2L interaction list
    //  *
    //  * This function regroups the entire update mechanism of the interaction
    //  * lists updating. It will dispatch the call to @ref update_interaction_list
    //  * which will update the list stored in the component's symbolics and also update
    //  * the iterator list according to the interaction list just updated.
    //  *
    //  * @param begin_of_source_groups first iterator on source group
    //  * @param end_of_source_groups  last iterator on source group
    //  * @param group the source group near or containing the Morton index
    //  * @param group_source_index index in vector of group of current source group of the component
    //  * @param component : The component to update its symbolics information.
    //  * @param level  the level in the tree of the component and the group
    //  * @param period : Array containing the periodicity of the box.
    //  * @param neighbour_separation The separation criteria
    //  */
    // template<typename GroupIterator, typename GroupS, typename GroupT, typename Component, typename Array>
    // inline auto build_interaction_list(GroupIterator begin_of_source_groups, GroupIterator end_of_source_groups,
    //                                    GroupS& group_source, std::size_t group_source_index, GroupT& group_target,
    //                                    Component& component, std::size_t level, Array const& period,
    //                                    const int& neighbour_separation, bool verbose = false) -> void
    // {
    //     // Build the theoretical interaction list.
    //     list::build_symbolic_interaction_list(operators::impl::tag_m2l{}, component, level, period,
    //                                           neighbour_separation);

    //     // Here we search the iterator for the multipoles (source cells)

    //     list::build_interaction_list_iterators(begin_of_source_groups, end_of_source_groups, group_target,
    //     group_source,
    //                                            group_source_index, component, verbose);
    // }
    /**
     * @brief Compute and set the interaction list for the group (morton indexes and iterators)
     *    This function regroups the entire update mechanism of the interaction
     * lists updating. It will dispatch the call to @ref update_interaction_list
     * which will update the list stored in the component's symbolics and also update
     * the iterator list according to the interaction list just updated.
     *
     * @tparam GroupIterator
     * @tparam Component
     * @tparam Array
     * @param group_target  the current target group containing cell
     * @param cell_target  the current target cell
     * @param begin_of_source_cell_groups  iterator on the first group of source cells
     * @param end_of_source_cell_groups   iterator on the last+1 group of source cells
     * @param level the level in the tree
     * @param period the vector of periodic conditions
     * @param neighbour_separation ste separation criterion
     */
    template<typename GroupT, typename GroupIteratorS, typename Component, typename Array>
    void build_m2l_interaction_list_for_group(GroupT& group_target, Component& cell_target,
                                              GroupIteratorS begin_of_source_cell_groups,
                                              GroupIteratorS end_of_source_cell_groups, std::size_t level,
                                              Array const& period, const int& neighbour_separation)
    {
        constexpr bool m2l_list = true;
        // find group_source containing the index of the target cell
        auto group_source =
          list::find_component_group_near(begin_of_source_cell_groups, end_of_source_cell_groups, cell_target.index());
        auto group_source_index = std::get<0>(group_source);
        auto group_source_it = std::get<1>(group_source);
	//        std::cout << " group_target " << group_target.csymbolics().idx_global << " group_source_index "
	//                  << group_source_index << std::endl;
        //
        // Build the theoretical interaction list.
        list::build_symbolic_interaction_list(operators::impl::tag_m2l{}, cell_target, level, period,
                                              neighbour_separation);

        // Here we search the iterator for the multipoles (source cells)

        list::build_interaction_list_iterators(begin_of_source_cell_groups, end_of_source_cell_groups, *group_source_it,
                                               group_source_index, group_target, cell_target, m2l_list);

        // list::build_interaction_list(begin_of_source_cell_groups, end_of_source_cell_groups, *group_source_it,
        //                              group_source_index, group_target, cell_target, level, period,
        //                              neighbour_separation, verbose);
    }
    ///
    ///               END M2L ROUTINES
    //////////////////////////////////////////////////////////////////////////////////
    ///               Routines to build the P2P interaction list
    ///
    /// @brief This function set the iterators list stored in the component symbolics for the p2p interaction list.
    ///
    /// This function updates 2 lists, one containing the iterators to the components
    /// inside the current group being processed, and another stored in the group symbolic infos
    /// that stores the out_of_block_interaction structure, allowing you to reconstruct mutual
    /// application of an operator like p2p.
    ///
    /// The method is optimized for mutual p2p operators. We only search iterators with Morton
    /// index smaller than Morton index of component
    /// Warning in the out_of_block list in the group we only have the indexes and not the iterators
    ///
    ///
    /// @param group : group of the component to update (target group)
    /// @param component : the component to update the iterator list
    /// @param component_index_in_group : The component index in its group.
    ///
    /// @return
    template<typename Group, typename Component>
    inline auto build_interaction_list_iterators(Group& group, Component& component,
                                                 int component_index_in_group) -> void
    {
        using group_type = Group;
        // Source group
        using out_of_block_interaction_type =
          typename scalfmm::component::symbolics_data<group_type>::out_of_block_interaction_type;
	//        static constexpr std::size_t number_of_interactions{
        //  scalfmm::component::symbolics_data<Component>::number_of_interactions};
        // Get symbolics infos on the group and the component
        auto& group_symbolics = group.symbolics();
        auto& component_symbolics = component.symbolics();
        auto idx_group = group_symbolics.idx_global;

        //  the list of interactions
        auto& interaction_indexes = component_symbolics.interaction_indexes;

        std::size_t counter_existing_component{0};
        int pos{0};
        // Loop on the number of neighbors
        const auto my_index = component.index();

        for(std::size_t index_interaction = 0; index_interaction < component_symbolics.number_of_neighbors; ++index_interaction)
        {
            // Get the morton index of the neighbor
            std::size_t current_interaction_morton_index =
              interaction_indexes.at(static_cast<std::size_t>(index_interaction));
            // We only search iterators with Morton index smaller than Morton index of component
            if(current_interaction_morton_index >= my_index)
            {
                continue;
            }
            // TODO : Put a function that explicitly specify the split
            // if the index is in the range of the group
            if(group_symbolics.starting_index <= current_interaction_morton_index &&
               current_interaction_morton_index < group_symbolics.ending_index)
            {
                // Get the position in group and check if the component exists
                const int leaf_position = group.component_index(current_interaction_morton_index);
                if(leaf_position != -1)
                {
                    component_symbolics.interaction_iterators.at(counter_existing_component) =
                      group.component_iterator(static_cast<std::size_t>(leaf_position));
                    // NEW
                    interaction_indexes.at(pos) = current_interaction_morton_index;
                    ++pos;
                    ++counter_existing_component;
                }
            }
            // if the index precedes the component, push an out of block interaction
            else if((current_interaction_morton_index < component_symbolics.morton_index) && (idx_group > 0))
            {
                out_of_block_interaction_type property(component_symbolics.morton_index,
                                                       current_interaction_morton_index, component_index_in_group);
                group_symbolics.outside_interactions.push_back(property);

            }
        }
        component_symbolics.existing_neighbors_in_group = counter_existing_component;

        if(group_symbolics.outside_interactions.size() > 0)
        {
            group.symbolics().outside_interactions_exists = true;
        }
    }
    /// @brief Calculates the range of outside interactions to compute
    /// according to the current group and the current outside interaction
    /// index.

    /// This function takes the current group symbolics in the loop
    /// and the group symbolics you have already processed. This means that,
    /// if you are processing the group 2 for outside interactions, you want
    /// get the range of interaction indices you need to processed in group 0 and
    /// group 1.
    ///
    /// @tparam GroupSymbolics: The group type
    /// @param current_group_symbolics : Current group symbolics information
    /// @param group_symbolics  The symbolics infos of the  group we search the index.
    /// @param current_out_interaction
    ///
    /// @return
    template<typename GroupSymbolics1, typename GroupSymbolics2>
    inline auto get_outside_interaction_range(GroupSymbolics1 const& current_group_symbolics,
                                              GroupSymbolics2 const& group_symbolics,
                                              std::size_t current_out_interaction)
    {
        const std::size_t block_start_index = group_symbolics.starting_index;
        const std::size_t block_end_index = group_symbolics.ending_index;
        //
        auto const& outside_interactions = current_group_symbolics.outside_interactions;
        // Advance until you reach the beginning of your previous group.
        while(current_out_interaction < outside_interactions.size() &&
              (outside_interactions.at(current_out_interaction).outside_index < block_start_index))
        {
            current_out_interaction++;
        }

        std::size_t last_out_interaction{current_out_interaction};
        // Get the last interaction you need to process in the previous group.
        //
        // WARNING block_end_index can be reach by leaves inside outside_interactions
        //  in a group block_end_index = last index + 1 ??
        while((last_out_interaction < outside_interactions.size()) &&
              outside_interactions.at(last_out_interaction).outside_index < block_end_index)
        {
            ++last_out_interaction;
        }

        return std::make_tuple(current_out_interaction, last_out_interaction);
    }
    /**
     * @brief Sort by group then by morton index inside a group the out of block interactions
     *
     * @tparam GroupIterator
     * @tparam Component
     * @param begin_of_groups The iterator on the set of groups
     * @param group the current group we treat
     */
   template<typename GroupIterator, typename Group>
    void sort_out_of_group_interactions(GroupIterator begin_of_groups, Group& group){

        auto& group_symbolics = group.symbolics();
       // const auto group_idx = group_symbolics.idx_global;
        // Get interactions outside of the current group.
        auto& outside_interactions = group_symbolics.outside_interactions;
            //   bool verbose = true;
        {
            // Sort the interactions to have continuous interactions in a group.
            std::sort(std::begin(outside_interactions), std::end(outside_interactions),
                      [](auto const& a, auto const& b) { return a.outside_index < b.outside_index; });
        }
        // if (verbose) {
        //      std::clog << cpp_tools::colors::green;
        //     std::clog << "out_of block: ";
        //     for(auto& u: outside_interactions)
        //     {
        //         std::clog << u << " ";
        //     }
        // }
        // std::clog << "\n" << cpp_tools::colors::reset;
        //
        // Sort the interactions inside a group to have continuous morton index.

        auto beg = std::begin(outside_interactions);
        auto current_group_iterator = begin_of_groups;
        const auto group_idx = group_symbolics.idx_global;

        std::advance(current_group_iterator, group_idx);
                std::size_t current_out_interaction{0};
        std::size_t first_out_interaction{0};
        std::size_t last_out_interaction{0};
        while(begin_of_groups != current_group_iterator && current_out_interaction < outside_interactions.size())
        {
            // get the interactions for the group (begin_of_groups) with the curent group
            //    and before the curent group
            std::tie(first_out_interaction, last_out_interaction) = list::get_outside_interaction_range(
              group_symbolics, (*begin_of_groups)->csymbolics(), current_out_interaction);
                std::sort(beg+first_out_interaction, beg+last_out_interaction,
                      [](auto const& a, auto const& b) {
                        return a.inside_index < b.inside_index;});

               ++begin_of_groups ;
        }
        // if (verbose) {
        //     std::cout << "out_of block: ";
        //     for(auto& u: outside_interactions)
        //     {
        //         std::clog << u << " ";
        //     }
        // }
        // std::clog << "\n";
        group_symbolics.outside_interactions_sorted = true;

    }
    /**
     * @brief Finalize the structure containing the out of block interactions of the group
     *
     * The goal is to check if the interaction with the outside leaf in the structure
     *   out_of_block_interaction exists and to find its position in its block
     *
     * Firstly for the current block we sort the interactions outside the group
     * according to the outside_index index (always bellow the first morton index of the group)
     *
     * Secondly, we iterate on the group before the group we are processing to apply the direct operator.
     * Also, for the group we extract the indices of this group and we apply to them the operator
     * p2p_mutual_apply.
     *
     * @todo improve the function group.component_index(block.outside_index) by using the previous find
     *
     * @tparam GroupIterator
     * @tparam MatrixKernel
     * @param group current block
     * @param begin_of_groups the iterator on the set of blocks
     */
    template<typename GroupIterator, typename Group>
    void build_out_of_group_interactions(GroupIterator begin_of_groups, /*scalfmm::component::group<Component>*/Group& group)
    {
        auto& group_symbolics = group.symbolics();
        const auto group_idx = group_symbolics.idx_global;
        // Get interactions outside of the current group.
        auto& outside_interactions = group_symbolics.outside_interactions;
        //
        // Sort the interactions to process the leaves in one block.
        sort_out_of_group_interactions(begin_of_groups, group);

        auto current_group_iterator = begin_of_groups;
        std::advance(current_group_iterator, group_idx);
        //
        std::size_t current_out_interaction{0};
        std::size_t first_out_interaction{0};
        std::size_t last_out_interaction{0};
        int pos{0};
        while(begin_of_groups != current_group_iterator && current_out_interaction < outside_interactions.size())
        {
            // Get for each previous group the indices range of interacting components and find its position
            // in its block
            // if(verbose)
            // {
            //     std::clog << "       it Group " << (*begin_of_groups)->csymbolics().idx_global << "  [ "
            //               << (*begin_of_groups)->csymbolics().starting_index << ", "
            //               << (*begin_of_groups)->csymbolics().ending_index<< "[ " << std::endl;
            // }
            std::tie(first_out_interaction, last_out_interaction) = list::get_outside_interaction_range(
              group_symbolics, (*begin_of_groups)->csymbolics(), current_out_interaction);
            {
                // #ifdef _OPENMP
                // Set the dependencies on the particles between previous groups
                if(last_out_interaction - first_out_interaction != 0)
                {
                    group_symbolics.group_dependencies.push_back((*begin_of_groups).get());
                }
                // #endif
                for(std::size_t out_inter_idx = first_out_interaction; out_inter_idx < last_out_interaction;
                    ++out_inter_idx)
                {
                    auto& block = outside_interactions.at(out_inter_idx);
                    // get the position (if it exists of the leaf) inside the group
                    // (costly, should be done once when the particles are fixed !!!)
                    const auto component_pos = (*begin_of_groups)->component_index(block.outside_index);

                    block.outside_index_in_block = component_pos;
                    if(component_pos != -1)
                    {
                        // better to add the iterator
                        auto cc = (*begin_of_groups)->begin();
                        std::advance(cc, block.outside_index_in_block);
                        block.outside_iterator = cc;
                        outside_interactions.at(pos) = block;
                        ++pos;
                    }
                }
            }
            current_out_interaction = last_out_interaction;
            ++begin_of_groups;
        }
        outside_interactions.resize(pos);

    }

    /// @defgroup update_interaction_list update_interaction_list
    ///

    /// @ingroup update_interaction_list
    /// @brief This function takes a leaf and its level in the tree
    /// and updates the component's symbolics information : the morton indices
    /// of its neighbors and the number of neighbors. It returns the theoretical
    /// interaction list of the leaf.
    ///
    /// @tparam[in] Tag : operator tag to select specific overload of @ref get_interaction_neighbors
    /// @param[inout] component : the component in which the list will be updated i.e its symbolics component
    /// @param[in] tree_level : the component's level in the tree
    /// @param[in] period array of periodicity in each direction
    /// @param[in] neighbour_separation distance between neighbors and component
    /// @param[in] source_target specify if the source leaf is equal to the target leaf
    ///

    template<typename Component, typename Array>
    inline auto build_symbolic_interaction_list(operators::impl::tag_p2p t, Component& component,
                                                std::size_t tree_level, Array const& period,
                                                const int& neighbour_separation, const bool source_target) -> void
    {
        // Get symbolics infos on the group and the component
        auto& component_symbolics = component.symbolics();
        // Calculate the coordinate of the component from the morton index
        auto coordinate{
          index::get_coordinate_from_morton_index<Component::dimension>(component_symbolics.morton_index)};
        // build component's symbolics information
        auto interaction_neighbors =
          index::get_interaction_neighbors(t, coordinate, tree_level, period, neighbour_separation);
        component_symbolics.interaction_indexes = std::get<0>(interaction_neighbors);
        component_symbolics.number_of_neighbors = std::get<1>(interaction_neighbors);
        if(!source_target)
        {
            // case when source != target
            auto& num1 = component_symbolics.interaction_indexes;
            num1[component_symbolics.number_of_neighbors] = component.index();
            component_symbolics.number_of_neighbors++;
            std::sort(std::begin(num1), std::begin(num1) + component_symbolics.number_of_neighbors);
        }
    }
    /**
     * @brief Build the interactions inside the group
     *
     * @tparam GroupIterator
     * @tparam GroupType
     * @tparam Component
     * @tparam Array
     * @param leaf  target leaf
     * @param begin_of_source_groups iterator on the first source groups
     * @param end_of_source_groups iterator on the ast source groups
     * @param group_target target group containing the leaf target
     * @param index_in_group index of group_target in the vector of target groups
     * @param leaf_level  Tle level of the leaf
     * @param mutual   if we consider mutual interactions or no
     * @param period  the vector of periodicity
     * @param neighbour_separation the separation criterion
     * @param source_target if sources = targets or not
     */
    template<typename GroupIterator, typename GroupType, typename Component, typename Array>
    void build_p2p_interaction_list_inside_group(Component& leaf, GroupIterator begin_of_source_groups,
                                                 GroupIterator end_of_source_groups, GroupType& group_target,
                                                 std::size_t& index_in_group, std::size_t leaf_level, const bool mutual,
                                                 Array const& period, const int& neighbour_separation,
                                                 const bool source_target)
    {
        constexpr bool m2l_list = false;

        scalfmm::list::build_symbolic_interaction_list(operators::impl::tag_p2p{}, leaf, leaf_level, period,
                                                       neighbour_separation, source_target);
        auto& leaf_symbolics = leaf.symbolics();

        // get the iterators of the real leaves in the P2P interaction list
        if((!mutual) || (!source_target))
        {
            auto mine_index = leaf.index();
            // like in source target there are no dependencies between the leaves
            auto group_source =
              list::find_component_group_near(begin_of_source_groups, end_of_source_groups, mine_index);
            auto group_source_index = std::get<0>(group_source);
            auto group_source_it = std::get<1>(group_source);
            list::build_interaction_list_iterators(begin_of_source_groups, end_of_source_groups, *group_source_it,
                                                   group_source_index, group_target, leaf, m2l_list);
            leaf_symbolics.existing_neighbors_in_group = leaf_symbolics.number_of_neighbors;
        }
        else
        {
            // classical algorithm in P2P (mutual interactions).
            // in this case the iterators of teh source and of the target are the same (No problem)
            if constexpr(std::is_same_v<typename GroupType::symbolics_type::iterator_source_type,
                                        typename GroupType::symbolics_type::iterator_type>)
            {
                list::build_interaction_list_iterators(group_target, leaf, index_in_group);
            }
        }
    }
    ///             End of routines to build the P2P interaction list
    //////////////////////////////////////////////////////////////////////////////////
    ///
    /**
     * @brief Reconstruct the p2p interaction list by adding the interaction between groups
     *
     * This function is just for debug purpose
     * @warning After this call the interaction list in the tree do not be used for an algorithm.
     * @param[inout] tree
     */
    template<typename TREE>
    void reconstruct_p2p_mutual_interaction_lists(TREE& tree)
    {
        component::for_each(std::get<0>(tree.begin()), std::get<0>(tree.end()),
                            [&tree](auto& group)
                            {
                                auto& group_symbolics = group->symbolics();
                                auto& out_of_group = group_symbolics.outside_interactions;

                                if(group_symbolics.outside_interactions_exists)
                                {
                                    auto& vect_leaf_updated = group->components();

                                    // the vector of interactions between groups exists

                                    //  we have to add interaction with inside morton index
                                    // loop on the group before me
                                    for(auto& u: out_of_group)
                                    {
                                        auto& leaf_symbolics = vect_leaf_updated[u.inside_index_in_block].symbolics();
                                        auto& indexes = leaf_symbolics.interaction_indexes;
                                        auto& iterators = leaf_symbolics.interaction_iterators;
                                        auto& nb = leaf_symbolics.existing_neighbors_in_group;
                                        indexes[nb] = u.outside_index;
                                        iterators[nb] = u.outside_iterator;
                                        ++nb;
                                    }
                                }
                            });
        // we  the interaction list of each leaf
        component::for_each_leaf(std::begin(tree), std::end(tree),
                                 [](auto& leaf)
                                 {
                                     auto& leaf_symbolics = leaf.symbolics();
                                     auto& indexes = leaf_symbolics.interaction_indexes;
                                     auto& iterators = leaf_symbolics.interaction_iterators;
                                     auto nb = leaf_symbolics.existing_neighbors_in_group;
                                     std::sort(std::begin(indexes), std::begin(indexes) + nb);
                                     std::sort(std::begin(iterators), std::begin(iterators) + nb,
                                               [](auto const& a, auto const& b) { return a->index() < b->index(); });
                                 });
    }
}   // namespace scalfmm::list

#endif   // SCALFMM_LISTS_UTIL_HPP
