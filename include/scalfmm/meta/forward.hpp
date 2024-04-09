// --------------------------------
// See LICENCE file at project root
// File : meta/forward.hpp
// --------------------------------
#ifndef SCALFMM_META_FORWARD_HPP
#define SCALFMM_META_FORWARD_HPP

// Forward declaration for traits support
#include <cstddef>

namespace scalfmm::container
{
    template<typename Derived, typename... Containers>
    struct variadic_adaptor;
    template<typename Derived, template<typename U, typename Allocator> class Container, typename... Types>
    struct unique_variadic_container;
    template<typename Derived, typename... Types>
    struct variadic_container;
    template<typename Derived, typename Tuple>
    struct variadic_container_tuple;
    template<typename ValueType, std::size_t Dimension>
    struct point_impl;
    template<typename ValueType, std::size_t Dimension>
    struct point_proxy;
    template<typename ValueType, std::size_t Dimension, typename Enable>
    struct point;
    template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
             typename OutputsType, std::size_t MOutputs, typename... Variables>
    struct particle;
}   // namespace scalfmm::container


#endif // SCALFMM_META_FORWARD_HPP

