// --------------------------------
// See LICENCE file at project root
// File : algorithm/sequential/transfert.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_TRANSFER_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_TRANSFER_HPP

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/m2l.hpp"
#include "scalfmm/operators/mutual_apply.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

#include <algorithm>
#include <array>
#include <ios>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace scalfmm::algorithms::sequential::pass
{
    /// @brief This function transfer the multipoles to the local
    /// expansions by applying the matrix kernel. This is
    /// performed by the m2l operator.
    ///
    /// @tparam Tree tree type
    /// @tparam Approximation interpolation type
    /// @param tree reference on the tree
    /// @param approximation const reference on the approximation
    ///
    // template<typename Tree, typename FarField>
    // inline auto transfer(Tree& tree, FarField const& far_field) -> void
    template<typename TreeSource, typename TreeTarget, typename FarField>
    inline auto transfer(TreeSource& tree_source, TreeTarget& tree_target, FarField const& far_field) -> void
    {
        // bool source_eq_target{true};
        // if(&tree_source != &tree_target)
        // {
        //     source_eq_target = false;
        // }
        bool source_eq_target{false};
        if constexpr(std::is_same_v<TreeSource, TreeTarget>)
        {
            source_eq_target = (&tree_source == &tree_target);
        }
        // auto& tree = tree_source;
        using operators::m2l_loop;   // Useful to overload m2l(...) function

        ///
        auto tree_height = tree_target.height();
        // Should be setted by the kernel !
        auto const& approximation = far_field.approximation();

        // Buffer for m2l optimization : we calculate the inverse fft on the fly and free memory.
        // get the shape of the buffer used in the m2l pass (get it from the far field)
        // and allocate it
        auto buffer(approximation.buffer_initialization());

        /////////////////////////////////////////////////////////////////////////////////////////
        ///  Loop on the level from the level 1/2 to the leaf level (Top to Bottom)
        //
        const auto top_height = tree_target.box().is_periodic() ? 1 : 2;
        //
        // auto begin_target = std::begin(tree_target);
        // auto end_target = std::end(tree_target);
        // //  auto cell_level_it = std::get<1>(begin_target) + top_height;   //
        // auto cell_target_level_it = std::get<1>(begin_target) + top_height;
        // auto cell_source_level_it = std::get<1>(std::begin(tree_source)) + top_height;
        // Iterate on the target tree
        for(std::size_t level = top_height; level < tree_height; ++level)
        {
            auto begin_cell_target_level_it = tree_target.begin_cells(level);
            auto end_cell_target_level_it = tree_target.end_cells(level);

            /////////////////////////////////////////////////////////////////////////////////////////
            ///            loop on the target groups at level level
            for(auto cell_target_level_it = begin_cell_target_level_it;
                cell_target_level_it != end_cell_target_level_it; ++cell_target_level_it)
            {
                auto group_target = cell_target_level_it->get();
                //
                /////////////////////////////////////////////////////////////////////////////////////////
                ///          loop on the target leaves of the current group
                for(std::size_t index_in_group{0}; index_in_group < group_target->size() /*std::size(*group_target)*/;
                    ++index_in_group)
                {
                    auto& target_cell = group_target->component(index_in_group);
                    auto const& cell_symbolics = target_cell.csymbolics();

                    m2l_loop(approximation, target_cell, cell_symbolics.level, buffer);

                    /// post-processing the leaf if necessary
                    approximation.apply_multipoles_postprocessing(target_cell, buffer);

                    approximation.buffer_reset(buffer);
                }
                /// post-processing the group if necessary
            }
        }
    }
}   // namespace scalfmm::algorithms::sequential::pass
#endif   // SCALFMM_ALGORITHMS_SEQUENTIAL_TRANSFER_HPP
