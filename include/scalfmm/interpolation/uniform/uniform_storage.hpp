// --------------------------------
// See LICENCE file at project root
// File : interpolation/uniform.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_STORAGE_HPP
#define SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_STORAGE_HPP

#include "scalfmm/container/variadic_adaptor.hpp"
#include "scalfmm/memory/storage.hpp"
#include <complex>

namespace scalfmm::component
{
    // This is the specialization for the uniform interpolator that
    // requires to store the transformed multipoles.
    template<typename ValueType, std::size_t Dimension, std::size_t Inputs, std::size_t Outputs>
    struct alignas(XTENSOR_FIXED_ALIGN) uniform_fft_storage
      : public memory::aggregate_storage<memory::multipoles_storage<ValueType, Dimension, Inputs>,
                                         memory::locals_storage<ValueType, Dimension, Outputs>,
                                         memory::transformed_multipoles_storage<std::complex<ValueType>, Dimension, Inputs>>
    {
        static constexpr std::size_t dimension = Dimension;
        static constexpr std::size_t inputs_size = Inputs;
        static constexpr std::size_t outputs_size = Outputs;

        using value_type = ValueType;
        using multipoles_storage_type = memory::multipoles_storage<ValueType, Dimension, Inputs>;
        using locals_storage_type = memory::locals_storage<ValueType, Dimension, Outputs>;
        using transformed_storage_type = memory::transformed_multipoles_storage<std::complex<ValueType>, Dimension, Inputs>;
        using base_type =
          memory::aggregate_storage<multipoles_storage_type, locals_storage_type, transformed_storage_type>;
        using multipoles_container_type = typename memory::storage_traits<multipoles_storage_type>::tensor_type;
        using locals_container_type = typename memory::storage_traits<locals_storage_type>::tensor_type;
        using buffer_storage_type = memory::transformed_multipoles_storage<std::complex<ValueType>, Dimension, Outputs>;
        using buffer_value_type = typename memory::storage_traits<buffer_storage_type>::value_type;
        using buffer_inner_type = typename memory::storage_traits<buffer_storage_type>::inner_type;
        using buffer_shape_type = typename memory::storage_traits<buffer_storage_type>::outer_shape;
        using buffer_type = typename memory::storage_traits<buffer_storage_type>::tensor_type;

        using base_type::base_type;

        uniform_fft_storage() = default;
        uniform_fft_storage(uniform_fft_storage const&) = default;
        uniform_fft_storage(uniform_fft_storage&&) noexcept = default;
        inline auto operator=(uniform_fft_storage const&) -> uniform_fft_storage& = default;
        inline auto operator=(uniform_fft_storage&&) noexcept -> uniform_fft_storage& = default;
        ~uniform_fft_storage() = default;
    };
}   // namespace scalfmm::component
#endif   // SCALFMM_INTERPOLATION_UNIFORM_UNIFORM_STORAGE_HPP
