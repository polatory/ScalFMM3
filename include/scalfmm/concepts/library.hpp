// --------------------------------
// See LICENCE file at project root
// File : concepts/library.hpp
// --------------------------------
#ifndef SCALFMM_CONCEPTS_LIBRARY_HPP
#define SCALFMM_CONCEPTS_LIBRARY_HPP

#include <type_traits>

namespace scalfmm::concepts
{
    // template<typename R, typename Enabler>
    // struct require_impl;

    // template<typename R>
    // struct require_impl<R, void>
    //{
    //    using type = R;
    //};

    // template<typename Return, typename... Ts>
    // struct require_check : require_impl<Return, std::void_t<Ts...>>
    //{
    //};

    // template<typename From, typename To>
    // using Convertible = std::enable_if_t<std::is_convertible_v<From, To>>;

    // template<typename T>
    // using Arithmetic = std::enable_if_t<std::is_arithmetic_v<T>>;

    // template<typename T>
    // using Integral = std::enable_if_t<std::is_integral_v<T>>;

    // template<bool Condition>
    // using If = std::enable_if_t<Condition>;

}   // end namespace scalfmm::concepts

//// Pseudo require macro
//#define requires(...)->typename ::concept ::require_check < __VA_ARGS__> ::type

#endif   // SCALFMM_CONCEPTS_LIBRARY_HPP
