// --------------------------------
// See LICENCE file at project root
// File : core/operators.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP
#define SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/interpolation/permutations.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/meta/utils.hpp"

namespace count_kernels
{
    namespace particles
    {
        struct empty_shape
        {
        };
        struct empty_inner
        {
            template<typename S>
            empty_inner(S, double){};
        };
        struct empty
        {
            empty() = default;
            empty(empty_shape, empty_inner){};
        };
        ///
        /// \brief The count_matrix_kernel struct
        ///
        /// For compatibility
        struct count_matrix_kernel
        {
            static constexpr std::size_t kn{1};
            static constexpr std::size_t km{1};
            static constexpr int separation_criterion{1};
        };
        ///
        /// \brief The count_near_field struct
        ///
        /// An implementation of the near field for the counter kernel
        struct count_near_field
        {
            using matrix_kernel_type = count_matrix_kernel;
            count_matrix_kernel _mat;
            bool _mutual;
            inline auto separation_criterion() const -> int { return matrix_kernel_type::separation_criterion; }
            inline count_matrix_kernel matrix_kernel() const { return _mat; };
            inline auto mutual() const -> bool { return _mutual; }

            count_near_field(const bool mutual)
              : _mutual(mutual)
            {
            }
        };

        template<std::size_t Dimension = 3>
        struct count_approximation
        {
            static constexpr std::size_t dimension{Dimension};
            using matrix_kernel_type = count_matrix_kernel;
            using storage_type = scalfmm::component::grid_storage<double, Dimension, 1, 1>;
            count_matrix_kernel mat;
            inline auto matrix_kernel() const noexcept -> count_matrix_kernel const& { return mat; }

            inline auto cell_width_extension() const noexcept { return 0.; }

            using buffer_type = empty;
            using buffer_shape_type = empty_shape;
            using buffer_inner_type = empty_inner;
            template<typename Cell>
            void apply_multipoles_preprocessing(Cell& /*current_cell*/, std::size_t t = 0) const
            {
            }
            template<typename Cell>
            void apply_multipoles_postprocessing(Cell& /*current_cell*/, buffer_type, std::size_t t = 0) const
            {
            }

            inline auto buffer_initialization() const -> buffer_type { return empty{}; }
            inline auto buffer_shape() const -> buffer_shape_type { return empty_shape{}; }
            inline auto buffer_reset(buffer_type) const -> void {}
        };

        ///
        /// \brief The count_interpolator struct
        ///
        /// An implementation of the far field for the counter kernel
        ///
        template<std::size_t Dimension = 3>
        struct count_far_field
        {
          private:
            count_approximation<Dimension> m_count_approximation;

          public:
            //      inline auto separation_criterion() const -> std::size_t { return 1; }
            using approximation_type = count_approximation<Dimension>;
            static constexpr bool compute_gradient = false;
            auto approximation() const -> approximation_type const& { return m_count_approximation; }
            inline auto separation_criterion() const noexcept -> int { return 1; }
        };

        template<typename Leaf, typename Cell, std::size_t Dimension>
        inline void p2m(count_far_field<Dimension> const& /*interp*/, Leaf const& source_leaf, Cell& target_cell)
        {
            auto& multipoles = target_cell.multipoles().at(0);
            multipoles += source_leaf.size();
        }

        template<typename Cell, std::size_t Dimension>
        inline void l2l(count_approximation<Dimension> const& /*interp*/, Cell const& parent_cell,
                        std::size_t /*child_index*/, Cell& child_cell, std::size_t tree_level = 2)
        {
            auto& child_locals = child_cell.locals().at(0);
            auto const& current_locals = parent_cell.clocals().at(0);
            child_locals += current_locals;
        }

        template<typename Cell, std::size_t Dimension>
        inline void m2m(count_approximation<Dimension> const& /*interp*/, Cell const& child_cell,
                        std::size_t /*child_index*/, Cell& parent_cell, std::size_t tree_level = 2)
        {
            auto& parent_multipoles = parent_cell.multipoles().at(0);
            auto const& child_multipoles = child_cell.cmultipoles().at(0);
            parent_multipoles += child_multipoles;
        }

        template<typename Cell, std::size_t Dimension>
        inline void m2l(count_approximation<Dimension> const& /*interp*/, Cell const& source_cell,
                        std::size_t /*neighbor_idx*/, Cell& target_cell, std::size_t /*tree_level*/,
                        typename count_approximation<Dimension>::buffer_type& /*buffer*/)
        {
            auto const& source_multipoles = source_cell.cmultipoles().at(0);
            auto& target_locals = target_cell.locals().at(0);
            target_locals += source_multipoles;
        }
        template<typename Cell, std::size_t Dimension>
        inline void m2l_loop(count_approximation<Dimension> const& /*interp*/, Cell& target_cell,
                             std::size_t /*tree_level*/,
                             typename count_approximation<Dimension>::buffer_type& /*buffer*/)
        {
            auto const& target_symb = target_cell.csymbolics();
            auto const& interaction_iterators = target_symb.interaction_iterators;
            auto& target_locals = target_cell.locals().at(0);
            for(std::size_t index{0}; index < target_symb.existing_neighbors; ++index)
            {
                auto const& source_cell = *interaction_iterators.at(index);
                auto const& source_multipoles = source_cell.cmultipoles().at(0);
                target_locals += source_multipoles;
            }
        }
        template<typename Cell, typename Leaf, std::size_t Dimension>
        inline void l2p(count_far_field<Dimension> const& /*fmm_operator*/, Cell const& source_cell, Leaf& target_leaf)
        {
            auto p = typename Leaf::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += *std::begin(source_cell.clocals().at(0));
        }

        template<typename Leaf, typename ContainerOfLeafIterator, typename ArrayT, typename ValueT>
        inline void p2p_full_mutual(count_matrix_kernel const& /*mat*/, Leaf& target_leaf,
                                    ContainerOfLeafIterator const& neighbors, const int& size, ArrayT const&,
                                    ValueT const&)
        {
            using value_type = typename Leaf::value_type;
            value_type nb_val{0};
            std::for_each(std::begin(neighbors), std::begin(neighbors) + size,
                          [&nb_val, &target_leaf](auto& leaf_r)
                          {
                              nb_val += static_cast<value_type>(leaf_r->size());
                              auto p_r = typename Leaf::proxy_type(*(leaf_r->begin()));
                              p_r.outputs()[0] += target_leaf.size();
                          });

            auto p = typename Leaf::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += nb_val;
        }

        template<bool MutualComputation = false, typename LEAF_T>
        inline void p2p_inner([[maybe_unused]] count_matrix_kernel const& mat, LEAF_T& target_leaf, [[maybe_unused]] const bool mutual = true)
        {
            using value_type = typename LEAF_T::value_type;

            if constexpr(MutualComputation)
            {
                {
                    std::cout << cpp_tools::colors::cyan;
                    std::cout << "\t --- P2P inner mutual ---" << std::endl;
                    std::cout << cpp_tools::colors::reset;
                }
            }
            else
            {
                {
                    std::cout << cpp_tools::colors::cyan;
                    std::cout << "\t --- P2P inner non mutual ---" << std::endl;
                    std::cout << cpp_tools::colors::reset;
                }
            }
            auto p = typename LEAF_T::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += static_cast<value_type>(target_leaf.size());
        }
        template<typename Leaf, typename ContainerOfLeafIterator, typename ArrayT, typename ValueT>
        inline void p2p_outer([[maybe_unused]] count_matrix_kernel const& mat, Leaf& target_leaf,
                              ContainerOfLeafIterator const& neighbors, ArrayT const&, ValueT const&)
        {
            using value_type = typename Leaf::value_type;
            value_type nb_val{0};
            std::for_each(std::begin(neighbors), std::begin(neighbors) + target_leaf.csymbolics().number_of_neighbors,
                          [&nb_val](auto const& leaf)
                          {
                              {
                                  nb_val += static_cast<value_type>(leaf->size());
                              }
                          });
            auto p = typename Leaf::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += nb_val;
        }
        template<typename Leaf, typename ContainerOfLeafIterator, typename ArrayT, typename ValueT>
        inline void p2p_outer([[maybe_unused]] count_matrix_kernel const& mat, Leaf& target_leaf,
                              ContainerOfLeafIterator const& neighbors, const int& number_of_neighbors, ArrayT const&,
                              ValueT const&)
        {
            using value_type = typename Leaf::value_type;
            value_type nb_val{0};
            std::for_each(std::begin(neighbors), std::begin(neighbors) + number_of_neighbors,
                          [&nb_val](auto const& leaf)
                          {
                              {
                                  nb_val += static_cast<value_type>(leaf->size());
                              }
                          });
            auto p = typename Leaf::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += nb_val;
        }
        template<typename TargetLeaf, typename SourceLeaf, typename ArrayT>
        inline void p2p_outer(count_matrix_kernel const&, TargetLeaf& target_leaf, SourceLeaf const& source_leaf,
                              [[maybe_unused]] ArrayT const& shift)
        {
            using value_type = typename TargetLeaf::value_type;
            auto p = typename TargetLeaf::proxy_type(*(target_leaf.begin()));
            p.outputs()[0] += static_cast<value_type>(source_leaf.size());
        }
    }   // namespace particles
}   // namespace count_kernels
namespace scalfmm::interpolation
{
    template<typename T>
    struct interpolator_traits;

    template<std::size_t Dimension>
    struct interpolator_traits<count_kernels::particles::count_approximation<Dimension>>
    {
        using matrix_kernel_type = count_kernels::particles::count_matrix_kernel;
        static constexpr std::size_t dimension = Dimension;
        static constexpr std::size_t kn = matrix_kernel_type::kn;
        static constexpr std::size_t km = matrix_kernel_type::km;
        static constexpr bool enable_symmetries = (dimension < 4) ? true : false;
        using storage_type = scalfmm::component::grid_storage<double, dimension, km, kn>;
        using buffer_type = count_kernels::particles::empty;
        using buffer_shape_type = count_kernels::particles::empty_shape;
        using buffer_inner_type = count_kernels::particles::empty_inner;
        using multipoles_container_type = typename storage_type::multipoles_container_type;
        using locals_container_type = typename storage_type::locals_container_type;
    };
}   // namespace scalfmm::interpolation

#endif   // SCALFMM_CORE_COUNT_KERNEL_HPP
