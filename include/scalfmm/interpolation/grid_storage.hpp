// --------------------------------
// See LICENCE file at project root
// File : interpolation/uniform.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_GRID_STORAGE_HPP
#define SCALFMM_INTERPOLATION_GRID_STORAGE_HPP

#include "scalfmm/memory/storage.hpp"

namespace scalfmm::component
{
    // This is the specialization for the uniform interpolator that
    // requires to store the transformed multipoles.
    template<typename ValueType, std::size_t Dimension, std::size_t Inputs, std::size_t Outputs>
    struct alignas(XTENSOR_FIXED_ALIGN) grid_storage
      : public memory::aggregate_storage<memory::multipoles_storage<ValueType, Dimension, Inputs>,
                                         memory::locals_storage<ValueType, Dimension, Outputs>>
    //  ,
    //  memory::transformed_multipoles_storage<ValueType, Dimension, 2>>
    {
        static constexpr std::size_t dimension = Dimension;
        static constexpr std::size_t inputs_size = Inputs;
        static constexpr std::size_t outputs_size = Outputs;

        using value_type = ValueType;

        using multipoles_storage_type = memory::multipoles_storage<ValueType, Dimension, Inputs>;
        using locals_storage_type = memory::locals_storage<ValueType, Dimension, Outputs>;
        // using buffer_storage_type = memory::tensor_storage<ValueType, Dimension, 2>;

        using base_type = memory::aggregate_storage<multipoles_storage_type, locals_storage_type>;

        using multipoles_container_type = typename memory::storage_traits<multipoles_storage_type>::tensor_type;
        using locals_container_type = typename memory::storage_traits<locals_storage_type>::tensor_type;

        using base_type::base_type;
        // more using in memory

        grid_storage() = default;
        grid_storage(grid_storage const&) = default;
        grid_storage(grid_storage&&) noexcept = default;
        inline auto operator=(grid_storage const&) -> grid_storage& = default;
        inline auto operator=(grid_storage&&) noexcept -> grid_storage& = default;
        ~grid_storage() = default;
    };
}   // namespace scalfmm::component
#endif   // SCALFMM_INTERPOLATION_GRID_STORAGE_HPP
