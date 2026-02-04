#pragma once

#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/operators/fmm_operators.hpp"

#include <cstddef>

namespace check_adl
{
    namespace operators
    {
        /// Empty structure (the kernel is not used )
        struct empty_kernel
        {
            // static constexpr std::size_t dimension = 3;
            // Mandatory constants
            static constexpr auto symmetry_tag{
              scalfmm::matrix_kernels::symmetry::non_symmetric};   // Specify the symmetry of the kernel.
            // static constexpr auto homogeneity_tag{
            //   scalfmm::matrix_kernels::homogeneity::non_homogenous};   // Specify the homogeneity of the kernel.
            static constexpr std::size_t km{1};   // The number of inputs of the kernel.
            static constexpr std::size_t kn{1};
            // The number of outputs of the kernel.
            static constexpr int separation_criterion{1};   // Criterion used to separate near and far field.
            //
            const std::string name() const { return std::string("empty_kernel"); }
            // #ifdef CHECK_ADL_ON_P2M
            //             template<typename ValueType>
            //             using vector_type = std::array<ValueType, kn>;   // Vector type that is used in the kernel.
            //             template<typename ValueType>
            //             [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
            //             {
            //                 return vector_type<ValueType>({ValueType(1.) / (cell_width * cell_width)});
            //             }
            // #endif
        };

#ifdef CHECK_ADL_ON_P2P
#ifdef CHECK_ADL_ON_P2P_WITH_USING
        using near_field_operator_type = scalfmm::operators::near_field_operator<check_adl::operators::empty_kernel>;
#warning ("Compile P2P with using");
#else
#warning ("Compile P2P with struct");

        struct near_field_operator_type : scalfmm::operators::near_field_operator<check_adl::operators::empty_kernel>
        {
            using base_type = scalfmm::operators::near_field_operator<check_adl::operators::empty_kernel>;
            using base_type::base_type;
        };
#endif
#endif

#ifdef CHECK_ADL_ON_P2M
#ifdef CHECK_ADL_ON_P2M_WITH_USING
#warning ("Compile P2M with using");

        template<typename Approximation, bool ComputeGradient = false>
        using new_far_field_operator = scalfmm::operators::far_field_operator<Approximation, ComputeGradient>;
#else
#warning ("Compile P2M with struct");

        template<typename Approximation, bool ComputeGradient = false>
        struct new_far_field_operator : scalfmm::operators::far_field_operator<Approximation, ComputeGradient>
        {
            using base_type = scalfmm::operators::far_field_operator<Approximation, ComputeGradient>;
            using base_type::base_type;
        };
#endif
#endif
    }   // namespace operators
}   // namespace check_adl

namespace check_adl::operators   // OK for P2P
{
    template<typename Leaf, typename ContainerOfLeafIterator, typename ArrayType, typename ValueType>
    inline void p2p_outer(check_adl::operators::empty_kernel const& matrix_kernel, Leaf& target_leaf,
                          ContainerOfLeafIterator const& neighbors, ArrayType const& pbc, ValueType const& box_width)
    {
        std::cout << " check_adl coucou p2p_outer \n ";
    }
    template<typename LeafType, typename ContainerOfLeafIteratorType, typename ArrayType, typename ValueType>
    inline void p2p_full_mutual([[maybe_unused]] check_adl::operators::empty_kernel const& matrix_kernel,
                                [[maybe_unused]] LeafType& target_leaf,
                                [[maybe_unused]] ContainerOfLeafIteratorType const& neighbors, const int& size,
                                [[maybe_unused]] ArrayType const& pbc, [[maybe_unused]] ValueType const& box_width)
    {
        throw std::runtime_error("No p2p_full_mutual for this kernel...\n");
    }
    template<typename Leaf>
    inline auto p2p_inner_mutual([[maybe_unused]] check_adl::operators::empty_kernel const& matrix_kernel,
                                 [[maybe_unused]] Leaf& target_leaf) -> void
    {
        throw std::runtime_error("No p2p_inner_mutual for this kernel...\n");
    }
    template<typename Leaf>
    inline auto p2p_inner(check_adl::operators::empty_kernel const& matrix_kernel, Leaf& target_leaf) -> void
    {
        std::cout << " check_adl coucou p2p_inner \n ";
    }
    template<typename Leaf>
    inline auto p2p_inner(check_adl::operators::empty_kernel const& matrix_kernel, Leaf& target_leaf,
                          [[maybe_unused]] const bool mutual) -> void
    {
        if(mutual)
        {
            throw std::runtime_error("No p2p_inner with mutual argument for this kernel...\n");
        }
        std::cout << " check_adl coucou p2p_inner with mutual \n ";
    }
    template<typename Leaf>
    inline auto p2p_inner_non_mutual(check_adl::operators::empty_kernel const& matrix_kernel, Leaf& target_leaf) -> void
    {
        std::cout << " check_adl coucou p2p_inner_non_mutual \n ";
    }
}   // namespace check_adl::operators

#ifdef CHECK_ADL_ON_P2M
namespace check_adl::operators
{
    template<typename Interpolator, bool ComputeGradient, typename CellType, typename LeafType>
    inline auto p2m(new_far_field_operator<Interpolator, ComputeGradient> const& far_field, LeafType const& leaf,
                    CellType& cell) -> void
    {
        std::cout << " my p2m with check_adl::operators::new_far_field_operator<...> \n ";
    }
}   // namespace check_adl::operators
#endif
//
