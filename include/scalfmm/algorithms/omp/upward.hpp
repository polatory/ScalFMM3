// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/upward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_UPWARD_HPP
#define SCALFMM_ALGORITHMS_OMP_UPWARD_HPP

#ifdef _OPENMP

#include <omp.h>

#include "scalfmm/operators/m2m.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

namespace scalfmm::algorithms::omp::pass
{

    /**
     * @brief perform the m2m operator for a given level
     *
     * @tparam Tree
     * @tparam Approximation
     * @param level current level to construct the m2m
     * @param tree the tree
     * @param approximation the approximation
     */
    template<typename Tree, typename Approximation>
    inline auto upward_level(const int& level, Tree& tree, Approximation const& approximation) -> void
    {
        using scalfmm::operators::m2m;
        //
        using interpolator_type = typename std::decay_t<Approximation>;
        static constexpr auto dimension{interpolator_type::dimension};
        static constexpr auto prio{omp::priorities::m2m};
        // Get the index of the corresponding child-parent interpolator
        std::size_t level_interpolator_index = (approximation.cell_width_extension() == 0.) ? 2 : level;
        //
        // iterator on the groups of cells (child level)
        auto group_of_child_cell_begin = tree.begin_cells(level + 1);
        auto group_of_child_cell_end = tree.end_cells(level + 1);
        // iterator on the groups of cells (current level)
        auto group_of_cell_begin = tree.begin_mine_cells(level);
        auto group_of_cell_end = tree.end_mine_cells(level);

        auto start_range_dependencies{group_of_child_cell_begin};
        auto end_range_dependencies{group_of_child_cell_begin};
        // We iterate on the parent cells
        while(group_of_cell_begin != group_of_cell_end)
        {
            using ptr_child_groups_type = std::decay_t<decltype(group_of_child_cell_begin->get())>;
            std::vector<ptr_child_groups_type> child_groups;

            auto group_parent = group_of_cell_begin->get();
            auto const& grp_parent_sym = group_parent->csymbolics();
            // pointer on the first multipole of the group
            auto group_parent_raw = &group_parent->ccomponent(0).cmultipoles(0);

            auto& child_dependencies{(*group_of_cell_begin)->symbolics().group_dependencies_m2m_in};

            std::tie(start_range_dependencies, end_range_dependencies) = index::get_child_group_range<dimension>(
              group_of_child_cell_begin, group_of_child_cell_end, *group_parent);

            while(start_range_dependencies != end_range_dependencies)
            {
                child_dependencies.push_back(&(*start_range_dependencies)->ccomponent(0).cmultipoles(0));
                child_groups.push_back(start_range_dependencies->get());
                ++start_range_dependencies;
            }

            start_range_dependencies = --end_range_dependencies;

#pragma omp task untied default(none) firstprivate(group_parent, child_groups)                                         \
  shared(approximation, level_interpolator_index) depend(iterator(std::size_t it = 0                                   \
                                                                  : grp_parent_sym.group_dependencies_m2m_in.size()),  \
                                                         in                                                            \
                                                         : (grp_parent_sym.group_dependencies_m2m_in.at(it))[0])       \
    depend(inout                                                                                                       \
           : group_parent_raw[0]) priority(prio)
            {   // Can be a task(in:iterParticles, out:iterChildCells ...)

                for(std::size_t cell_index = 0; cell_index < group_parent->size(); ++cell_index)
                {
                    auto& parent_cell = group_parent->component(cell_index);
                    auto parent_morton_index = parent_cell.index();
                    static constexpr auto number_of_child = math::pow(2, dimension);

                    for(auto p: child_groups)
                    {
                        for(auto const& child_cell: p->components())
                        {
                            auto child_morton_index{child_cell.index()};
                            if((child_morton_index >> dimension) == parent_morton_index)
                            {
                                const std::size_t child_index = child_morton_index & (number_of_child - 1);
                                m2m(approximation, child_cell, child_index, parent_cell, level_interpolator_index);
                            }
                        }
                    }
                    approximation.apply_multipoles_preprocessing(parent_cell, omp_get_thread_num());
                }
            }   // end pragma
            ++group_of_cell_begin;
        }
        assert(group_of_cell_begin == group_of_cell_end);
    }
    /// @brief This function constructs the local approximation for all the cells of the tree by applying the
    /// operator m2m
    ///
    /// @param tree   the tree target
    /// @param approximation the approximation to construct the local approximation
    ///
    template<typename Tree, typename Approximation>
    inline auto upward(Tree& tree, Approximation const& approximation) -> void
    {
        auto leaf_level = tree.height() - 1;
        //
        // upper working level is
        const int top_height = tree.box().is_periodic() ? 0 : 2;
        for(int level = leaf_level - 1; level >= top_height; --level)   // int because top_height could be 0
        {
            upward_level(level, tree, approximation);
        }
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_UPWARD_HPP
