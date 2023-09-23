#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_UTILS_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_UTILS_HPP

#include "scalfmm/tree/group_tree_view.hpp"

namespace scalfmm::algorithms::sequential
{

    /// @brief reset to zero the particles in the tree
    ///
    template<typename Tree>
    inline auto reset_particles(Tree& tree)
    {
        for(auto pg: tree.group_of_leaves())
        {
            // loop on leaves
            for(auto& leaf: pg->block())
            {
                leaf.particles().clear();
            }
        }
    }
    ///
    /// @brief reset to zero the output particles in the tree
    ///
    template<typename Tree>
    inline auto reset_outputs(Tree& tree)
    {
        // loop on group of leaves
        for(auto pg: tree.vector_of_leaf_groups())
        {
            // reset the output in the block
            pg->storage().reset_outputs();
        }
    }
    ///
    /// @brief reset to zero the multipole and the local values in the cells
    ///
    template<typename Tree>
    inline void reset_far_field(Tree& tree)
    {
        tree.reset_far_field();
    }
    ///
    /// @brief reset to zero the multipole values in the cells
    ///
    template<typename Tree>
    inline void reset_multipoles(Tree& tree)
    {
        tree.reset_multipoles();
    }
    ///
    /// @brief reset to zero the local values in the cells
    ///
    template<typename Tree>
    inline void reset_locals(Tree& tree)
    {
        tree.reset_locals();
    }
}   // namespace scalfmm::algorithms::sequential

#endif
