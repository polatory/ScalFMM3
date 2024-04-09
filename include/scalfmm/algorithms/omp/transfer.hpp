// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/transfert.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_OMP_TRANSFER_HPP
#define SCALFMM_ALGORITHMS_OMP_TRANSFER_HPP

#ifdef _OPENMP

#include <omp.h>
#include <vector>
#include <string>

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/m2l.hpp"
#include "scalfmm/operators/mutual_apply.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

namespace scalfmm::algorithms::omp::pass
{
    enum class split_m2l
    {
        full,                ///<  apply the transfer on all level of the tree
        remove_leaf_level,   ///<  apply the transfer on all level except the leaf level
        leaf_level           ///<  apply the transfer only on leaf level
    };

    template<typename TreeS, typename TreeT, typename FarField, typename BufferPtr>
    inline auto transfer_level(const int level, TreeS& tree_source, TreeT& tree_target, FarField const& far_field,
                               std::vector<BufferPtr> const& buffers) -> void
    {
        using operators::m2l_loop;   // Useful to overload m2l(...) function
        auto const& approximation = far_field.approximation();
        //
        auto begin_cell_target_level_it = tree_target.begin_cells(level);
        auto end_cell_target_level_it = tree_target.end_cells(level);
        auto num{0};
        /////////////////////////////////////////////////////////////////////////////////////////
        ///            loop on the groups at level level
        for(auto cell_target_level_it = begin_cell_target_level_it; cell_target_level_it != end_cell_target_level_it;
            ++cell_target_level_it,++num)
        {
            // get() because cell_target_level_it is a shared pointer
            auto group_ptr = cell_target_level_it->get();
            // dependence on the first local of the group
            auto ptr_local_raw = &group_ptr->ccomponent(0).clocals(0);
            static constexpr auto prio{priorities::m2l};
//
#pragma omp task untied default(none) firstprivate(group_ptr, ptr_local_raw) shared(approximation, buffers)            \
  depend(iterator(it = 0                                                                                               \
                  : std::size(group_ptr->csymbolics().group_dependencies_m2l)),                                        \
         in                                                                                                            \
         : (group_ptr->csymbolics().group_dependencies_m2l.at(it)[0]))     depend(inout                                \
           : ptr_local_raw[0]) priority(prio)
            {
                const auto thread_id{omp_get_thread_num()};
                ///////////////////////////////////////////////////////////////////////////////////////
                //          loop on the leaves of the current group
                for(std::size_t index_in_group{0}; index_in_group < std::size(*group_ptr); ++index_in_group)
                {
                    auto& target_cell = group_ptr->component(index_in_group);
                    auto const& cell_symbolics = target_cell.csymbolics();

                    m2l_loop(approximation, target_cell, cell_symbolics.level, *buffers.at(thread_id));

                    // post-processing the leaf if necessary
                    approximation.apply_multipoles_postprocessing(target_cell, *buffers.at(thread_id), thread_id);
                    approximation.buffer_reset(*buffers.at(thread_id));
                }
            }   // end pragma
                /// post-processing the group if necessary
        }
    }
    /// @brief apply the transfer operator to construct the local approximation in tree_target
    ///
    /// @tparam TreeS template for the Tree source type
    /// @tparam TreeT template for the Tree target type
    /// @tparam FarField template for the far field type
    /// @tparam BufferPtr template for the type of pointer of the buffer
    /// @param tree_source the tree containing the source cells/leaves
    /// @param tree_target the tree containing the target cells/leaves
    /// @param far_field The far field operator
    /// @param buffers vector of buffers used by the far_field in the transfer pass (if needed)
    /// @param split the enum  (@see split_m2l) tp specify on which level we apply the transfer
    /// operator
    ///
    template<typename TreeS, typename TreeT, typename FarField, typename BufferPtr>
    inline auto transfer(TreeS& tree_source, TreeT& tree_target, FarField const& far_field,
                         std::vector<BufferPtr> const& buffers, split_m2l split = split_m2l::full) -> void
    {
        //
        auto tree_height = tree_target.height();

        /////////////////////////////////////////////////////////////////////////////////////////
        ///  Loop on the level from the level top_height to the leaf level (Top to Bottom)
        auto top_height = tree_target.box().is_periodic() ? 1 : 2;
        auto last_level = tree_height;

        switch(split)
        {
        case split_m2l::remove_leaf_level:
            last_level--;
            ;
            break;
        case split_m2l::leaf_level:
            top_height = (tree_height - 1);
            break;
        case split_m2l::full:
            break;
        }
        for(std::size_t level = top_height; level < last_level; ++level)
        {
            transfer_level(level, tree_source, tree_target, far_field, buffers);
        }
    }
}   // namespace scalfmm::algorithms::omp::pass

#endif   // _OPENMP
#endif   // SCALFMM_ALGORITHMS_OMP_TRANSFER_HPP
