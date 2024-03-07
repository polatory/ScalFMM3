#ifndef SCALFMM_ALGORITHMS_OPENMP_UTILS_HPP
#define SCALFMM_ALGORITHMS_OPENMP_UTILS_HPP

namespace scalfmm::algorithms::omp
{
    namespace impl
    {
        /// @brief
        ///
        /// @tparam Approximation
        /// @param approximation
        ///
        /// @return
        template<typename Approximation>
        auto init_buffers(Approximation const& approximation)
          -> std::vector<std::decay_t<decltype(std::declval<Approximation>().buffer_initialization())>*>
        {
            using buffer_type = typename Approximation::buffer_type;
            using ptr_buffer_type = buffer_type*;

            std::vector<ptr_buffer_type> buffers{};

#pragma omp parallel
            {
#pragma omp single
                {
                    buffers.resize(omp_get_num_threads());
                }
                buffers.at(omp_get_thread_num()) = new buffer_type(approximation.buffer_initialization());
            }

            return buffers;
        }

        template<typename BuffersPtr>
        auto delete_buffers(std::vector<BuffersPtr> const& buffers) -> void
        {
#pragma omp parallel
            {
                delete buffers.at(omp_get_thread_num());
            }
        }

    }   // namespace impl
    /// @brief reset to zero the particles in the tree
    ///
    template<typename Tree>
    inline auto reset_particles(Tree& tree)
    {
#pragma omp parallel shared(tree)
#pragma omp single nowait
        {
            for(auto pg: tree.group_of_leaves())
            {
#pragma omp task firstprivate(pg)
                // loop on leaves
                for(auto& leaf: pg->block())
                {
                    leaf.particles().clear();
                }
            }
        }
#pragma omp taskwait
    }
    ///
    /// @brief reset to zero the output particles in the tree
    ///
    template<typename Tree>
    inline auto reset_outputs(Tree& tree)
    {
        // loop on group of leaves
        // No openmp
        // #pragma omp parallel shared(tree)
        // #pragma omp single nowait
        {
            // #pragma omp parallel for shared(tree)
            for(auto pg: tree.group_of_leaves())
            {
                // #pragma omp task firstprivate(pg)
                {
                    // reset the output in the block
                    pg->storage().reset_outputs();
                }
            }
            // #pragma omp taskwait
        }
    }
    ///
    /// @brief reset to zero the multipole and the local values in the cells
    ///
    template<typename Tree>
    inline void reset_far_field(Tree& tree)
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                auto cell_level_it = tree.cbegin_cells() + (tree.height() - 1);

                int top_level = tree.box().is_periodic() ? 0 : 2;
                for(int level = int(tree.height()) - 1; level >= top_level; --level)
                {
                    auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                    auto group_of_cell_end = std::cend(*(cell_level_it));
                    for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it)
                    {
                        auto const& ptr_group = *it;
#pragma omp task firstprivate(ptr_group)
                        {
                            // auto const& current_group_symbolics = ptr_group->csymbolics();
                            component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                [](auto& cell)
                                                {
                                                    cell.reset_multipoles();
                                                    cell.reset_locals();
                                                });
                        }
                    }
                    --cell_level_it;
                }
            }
#pragma omp taskwait
        }   // namespace scalfmm::algorithms::omp
    }

    ///
    /// @brief reset to zero the multipole values in the cells
    ///
    template<typename Tree>
    inline void reset_multipoles(Tree& tree)
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                auto cell_level_it = tree.cbegin_cells() + (tree.height() - 1);

                int top_level = tree.box().is_periodic() ? 0 : 2;
                for(int level = int(tree.height()) - 1; level >= top_level; --level)
                {
                    auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                    auto group_of_cell_end = std::cend(*(cell_level_it));
                    for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it)
                    {
                        auto const& ptr_group = *it;
#pragma omp task firstprivate(ptr_group)
                        {
                            // auto const& current_group_symbolics = ptr_group->csymbolics();
                            component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                [](auto& cell) { cell.reset_multipoles(); });
                        }
                    }
                    --cell_level_it;
                }
            }
#pragma omp taskwait
        }
    }
    ///
    /// @brief reset to zero the local values in the cells
    ///
    template<typename Tree>
    inline void reset_locals(Tree& tree)
    {
#pragma omp parallel shared(tree)
        {
#pragma omp single nowait
            {
                auto cell_level_it = tree.cbegin_cells() + (tree.height() - 1);

                int top_level = tree.box().is_periodic() ? 0 : 2;
                for(int level = int(tree.height()) - 1; level >= top_level; --level)
                {
                    auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                    auto group_of_cell_end = std::cend(*(cell_level_it));
                    for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it)
                    {
                        auto const& ptr_group = *it;
#pragma omp task firstprivate(ptr_group)
                        {
                            // auto const& current_group_symbolics = ptr_group->csymbolics();
                            component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                [](auto& cell) { cell.reset_locals(); });
                        }
                    }
                    --cell_level_it;
                }
            }
#pragma omp taskwait
        }
    }
}   // namespace scalfmm::algorithms::omp

#endif
