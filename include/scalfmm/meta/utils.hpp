// --------------------------------
// See LICENCE file at project root
// File : meta/utils.hpp
// --------------------------------
#ifndef SCALFMM_META_UTILS_HPP
#define SCALFMM_META_UTILS_HPP

#include <algorithm>
#include <any>
#include <array>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <xtensor/xtensor_forward.hpp>

#include "scalfmm/meta/forward.hpp"
#include "scalfmm/meta/traits.hpp"
#include "xtensor/xlayout.hpp"

namespace scalfmm
{
    namespace container
    {
        template<typename Derived, template<typename U, typename Allocator> class Container, typename... Types>
        struct unique_variadic_container;
        template<typename Derived, typename... Containers>
        struct variadic_adaptor;
        template<typename Derived, typename... Types>
        struct variadic_container;
        template<typename Derived, typename Tuple>
        struct variadic_container_tuple;
        template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
                 typename OutputsType, std::size_t MOutputs, typename... Variables>
        struct particle;
        template<typename ValueType, std::size_t Dimension, typename Enable>
        struct point;
        template<typename ValueType, std::size_t Dimension>
        struct point_impl;
        template<typename ValueType, std::size_t Dimension>
        struct point_proxy;
    }   // namespace container
}   // namespace scalfmm

namespace scalfmm::meta
{
    struct noop_t
    {
        template<typename... Types>
        noop_t(const Types&...)
        {
        }
    };

    struct noop_f
    {
        template<typename... F>
        noop_f(F...)
        {
        }
    };

    template<typename, typename>
    struct cat;

    template<typename... L, typename... R>
    struct cat<std::tuple<L...>, std::tuple<R...>>
    {
        using type = std::tuple<L..., R...>;
    };

    // standard traits and forward declaration
    template<typename T>
    struct tuple_size : std::tuple_size<T>
    {
    };

    template<typename T>
    static constexpr std::size_t tuple_size_v = meta::tuple_size<T>::value;

    template<typename ET, typename S, xt::layout_type L, bool SH>
    struct tuple_size<xt::xtensor_fixed<ET, S, L, SH>>
      : std::tuple_size<typename xt::xtensor_fixed<ET, S, L, SH>::inner_shape_type>
    {
    };

    template<typename Derived, typename... Containers>
    struct tuple_size<container::variadic_adaptor<Derived, Containers...>>
      : tuple_size<typename container::variadic_adaptor<Derived, Containers...>::base_type>
    {
    };

    template<typename Derived, template<typename U, typename Allocator> class Container, typename... Types>
    struct tuple_size<container::unique_variadic_container<Derived, Container, Types...>>
      : tuple_size<typename container::unique_variadic_container<Derived, Container, Types...>::base_type>
    {
    };

    template<typename Derived, typename... Types>
    struct tuple_size<container::variadic_container<Derived, Types...>>
      : tuple_size<typename container::variadic_container<Derived, Types...>::base_type>
    {
    };

    template<typename Derived, typename Tuple>
    struct tuple_size<container::variadic_container_tuple<Derived, Tuple>>
      : tuple_size<typename container::variadic_container_tuple<Derived, Tuple>::base_type>
    {
    };

    template<typename ValueType, std::size_t Dimension>
    struct tuple_size<scalfmm::container::point_impl<ValueType, Dimension>>
      : tuple_size<typename scalfmm::container::point_impl<ValueType, Dimension>::base_type>
    {
    };

    template<typename ValueType, std::size_t Dimension>
    struct tuple_size<scalfmm::container::point_proxy<ValueType, Dimension>>
      : tuple_size<typename scalfmm::container::point_proxy<ValueType, Dimension>::base_type>
    {
    };

    template<typename ValueType, std::size_t Dimension, typename Enable>
    struct tuple_size<scalfmm::container::point<ValueType, Dimension, Enable>>
      : tuple_size<typename scalfmm::container::point<ValueType, Dimension, Enable>::base_type>
    {
    };

    template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
             typename OutputsType, std::size_t MOutputs, typename... Variables>
    struct tuple_size<
      scalfmm::container::particle<PositionType, PositionDim, InputsType, NInputs, OutputsType, MOutputs, Variables...>>
      : std::integral_constant<std::size_t, PositionDim + NInputs + MOutputs + sizeof...(Variables)>
    {
    };

    template<std::size_t I, typename T>
    inline constexpr auto get(T&& t) noexcept -> auto&&
    {
        return std::get<I>(std::forward<T>(t));
    }

    template<typename ET, typename S, xt::layout_type L, bool SH, std::size_t... Is>
    inline constexpr auto get(xt::xtensor_fixed<ET, S, L, SH>&& exp) noexcept -> auto&&
    {
        return std::forward<xt::xtensor_fixed<ET, S, L, SH>>(exp).at(Is...);
    }

    template<std::size_t I, typename ValueType, std::size_t Dimension, typename Enable>
    inline constexpr auto get(container::point<ValueType, Dimension, Enable> const& p) noexcept -> ValueType const&
    {
        return p.at(I);
    }

    template<std::size_t I, typename ValueType, std::size_t Dimension, typename Enable>
    inline constexpr auto get(container::point<ValueType, Dimension, Enable>& p) noexcept -> ValueType&
    {
        return p.at(I);
    }

    //////////////////////////////////////////////
    template<std::size_t Added, std::size_t... Is>
    inline constexpr auto add_to_sequence(std::index_sequence<Is...> seq)
    {
        return std::index_sequence<(Added + Is)...>{};
    }

    template<typename T, size_t... I>
    inline constexpr auto reverse_impl(T&& t, std::index_sequence<I...> /*unused*/)
      -> std::tuple<typename std::tuple_element<sizeof...(I) - 1 - I, T>::type...>
    {
        return std::make_tuple(meta::get<sizeof...(I) - 1 - I>(std::forward<T>(t))...);
    }

    template<typename T>
    inline constexpr auto reverse(T&& t)
      -> decltype(reverse_impl(std::forward<T>(t),
                               std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>()))
    {
        return reverse_impl(std::forward<T>(t), std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>());
    }

    // Generate a tuple of Ts
    template<std::size_t, typename T>
    using type_id = T;

    template<typename T, std::size_t... Is>
    inline constexpr auto generate_tuple(std::index_sequence<Is...>)
    {
        return std::tuple<type_id<Is, T>...>{};
    }

    template<typename T, std::size_t N>
    inline constexpr auto generate_tuple()
    {
        return generate_tuple<T>(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    using generate_tuple_t = decltype(generate_tuple<T, N>());

    // template<template<class> class New, typename Tuple, std::size_t... Is>
    // inline constexpr auto replace_inner_tuple_type(std::index_sequence<Is...>/*s*/)
    //{
    //     return std::tuple<type_id<Is, New<typename std::tuple_element<Is, Tuple>::type>>...>{};
    // }

    template<template<class> class New, typename T>
    struct replace_inner_tuple_type;

    template<template<class> class New, typename... Ts>
    struct replace_inner_tuple_type<New, std::tuple<Ts...>>
    {
        using type = std::tuple<New<Ts>...>;
    };

    template<template<class> class New, typename T, std::size_t N>
    struct replace_inner_tuple_type<New, std::array<T, N>>
    {
        using type = std::array<New<T>, N>;
    };

    template<template<class> class New, typename T>
    using replace_inner_tuple_type_t = typename replace_inner_tuple_type<New, T>::type;

    //template<template<class> class New, typename Tuple>
    //inline constexpr auto replace_inner_tuple_type()
    //{
    //    return replace_inner_tuple_type<New, Tuple>(std::make_index_sequence<meta::tuple_size_v<Tuple>>{});
    //}

    //template<template<class> class New, typename Tuple>
    //using replace_inner_tuple_type_t = decltype(replace_inner_tuple_type<New, Tuple>());
    /**
     * @brief Transform a tuple in an array
     *
     * @param t the tuple - all elements are in the same type
     * @return constexpr auto
     */
    template<class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
    constexpr inline auto to_array(Tuple t)
    {
        return std::apply([](auto... elems) { return std::array<T, sizeof...(elems)>{elems...}; }, t);
    }
    constexpr inline auto to_array(std::tuple<>) { return std::array<double, 0>{}; }
    /**
     * @brief Transform an array in a tuple
     *
     * @param t the array
     * @return constexpr auto
     */
    template<class T, std::size_t N>
    constexpr inline auto to_tuple(std::array<T, N> a)
    {
        return std::apply([](auto... elems) { return std::make_tuple(elems...); }, a);
    }

    template<std::size_t Offset, typename Seq>
    struct offset_sequence;

    template<std::size_t Offset, std::size_t... Ints>
    struct offset_sequence<Offset, std::index_sequence<Ints...>>
    {
        using type = std::index_sequence<Offset + Ints...>;
    };

    template<std::size_t Offset, typename Seq>
    using offset_sequence_t = typename offset_sequence<Offset, Seq>::type;

    template<std::size_t Begin, std::size_t End>
    struct make_range_sequence : public offset_sequence_t<Begin, std::make_index_sequence<End - Begin>>
    {
    };

    template<typename T, std::size_t... Is>
    constexpr inline auto sub_tuple(T&& t, std::index_sequence<Is...>)
    {
        return std::forward_as_tuple(meta::get<Is>(std::forward<T>(t))...);
    }
    /**
     * @brief  extract a sub-tuple of the tuple according to the sequence of indexes
     *
      *
     * @param t  the tuple
     * @param s  the sequence of indexes
     * @return constexpr auto
        * \code {c++}
     *   // outputs_iterator is an iterator on the outputs (potential, forces)
     *    using range_pot = meta::make_range_sequence<0, 1>;
     *    auto & pot = meta::sub_tuple(outputs_iterator, range_pot{})
     * \endcode

     */
    template<typename T, std::size_t... Is>
    constexpr inline auto make_sub_tuple(T t, std::index_sequence<Is...> /*s*/)
    {
        return std::make_tuple(meta::get<Is>(t)...);
    }

    template<typename... Args>
    inline constexpr auto multiply(Args... args)
    {
        return (args * ...);
    }

    template<typename... Args>
    inline constexpr auto all(Args... args) -> bool
    {
        return (... && args);
    }

    namespace details
    {
        template<typename T, typename U, std::size_t... Is>
        inline constexpr auto tuple_sum_update(T&& lhs, U&& rhs, std::index_sequence<Is...>) -> void
        {
            noop_t{((meta::get<Is>(std::forward<T>(lhs)) += meta::get<Is>(std::forward<U>(rhs))), 0)...};
        }

        template<typename ArrayOfIt, typename TupleLike, std::size_t... Is>
        inline constexpr auto it_sum_update(ArrayOfIt&& lhs, TupleLike&& rhs, std::index_sequence<Is...>) -> void
        {
            noop_t{
              ((*meta::get<Is>(std::forward<ArrayOfIt>(lhs)) += meta::get<Is>(std::forward<TupleLike>(rhs))), 0)...};
        }

        template<typename T, typename U, std::size_t... Is>
        inline constexpr auto tuple_min_update(T&& lhs, U&& rhs, std::index_sequence<Is...>) -> void
        {
            noop_t{((meta::get<Is>(std::forward<T>(lhs)) -= meta::get<Is>(std::forward<U>(rhs))), 0)...};
        }

        template<typename F, typename LHS, typename RHS1, typename RHS2, std::size_t... Is>
        inline constexpr auto for_each(std::index_sequence<Is...> s, F&& f, LHS&& lhs, RHS1&& rhs1, RHS2&& rhs2) -> void
        {
            noop_t{(((meta::get<Is>(std::forward<LHS>(lhs)) =
                        std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<RHS1>(rhs1)),
                                    meta::get<Is>(std::forward<RHS2>(rhs2))))),
                    0)...};
        }

        template<typename T, typename U, typename F, std::size_t... Is>
        inline constexpr auto for_each(T&& lhs, U&& rhs, F&& f, std::index_sequence<Is...>) -> void
        {
            noop_t{((meta::get<Is>(std::forward<T>(lhs)) =
                       std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<U>(rhs)))),
                    0)...};
        }

        template<typename T, typename F, std::size_t... Is>
        inline constexpr auto for_each(T&& t, F&& f, std::index_sequence<Is...>) -> void
        {
            noop_t{(std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<T>(t))), 0)...};
        }

        template<typename F, typename T0, typename T1, typename T2, typename T3, std::size_t... Is>
        inline constexpr auto repeat(F&& f, T0&& t0, T1&& t1, T2&& t2, T3&& t3, std::index_sequence<Is...>) -> void
        {
            noop_t{(
              (std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<T0>(t0)), meta::get<Is>(std::forward<T1>(t1)),
                           meta::get<Is>(std::forward<T2>(t2)), meta::get<Is>(std::forward<T3>(t3)))),
              0)...};
        }

        template<typename F, typename T0, typename T1, typename T2, std::size_t... Is>
        inline constexpr auto repeat(F&& f, T0&& t0, T1&& t1, T2&& t2, std::index_sequence<Is...>) -> void
        {
            noop_t{((std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<T0>(t0)),
                                 meta::get<Is>(std::forward<T1>(t1)), meta::get<Is>(std::forward<T2>(t2)))),
                    0)...};
        }

        template<typename F, typename T0, typename T1, std::size_t... Is>
        inline constexpr auto repeat(F&& f, T0&& t0, T1&& t1, std::index_sequence<Is...>) -> void
        {
            noop_t{((std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<T0>(t0)),
                                 meta::get<Is>(std::forward<T1>(t1)))),
                    0)...};
        }

        template<typename F, typename T0, std::size_t... Is>
        inline constexpr auto repeat(F&& f, T0&& t0, std::index_sequence<Is...>) -> void
        {
            noop_t{((std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<T0>(t0)))), 0)...};
        }

        template<typename F, typename T0, typename T1, typename T2, typename T3, std::size_t... Is>
        inline constexpr auto repeat_indexed(F&& f, T0&& t0, T1&& t1, T2&& t2, T3&& t3,
                                             std::index_sequence<Is...> /*s*/) -> void
        {
            noop_t{((std::invoke(std::forward<F>(f), Is, meta::get<Is>(std::forward<T0>(t0)),
                                 meta::get<Is>(std::forward<T1>(t1)), meta::get<Is>(std::forward<T2>(t2)),
                                 meta::get<Is>(std::forward<T3>(t3)))),
                    0)...};
        }

        template<typename F, typename T0, typename T1, typename T2, std::size_t... Is>
        inline constexpr auto repeat_indexed(F&& f, T0&& t0, T1&& t1, T2&& t2, std::index_sequence<Is...>) -> void
        {
            noop_t{((std::invoke(std::forward<F>(f), Is, meta::get<Is>(std::forward<T0>(t0)),
                                 meta::get<Is>(std::forward<T1>(t1)), meta::get<Is>(std::forward<T2>(t2)))),
                    0)...};
        }

        template<typename F, typename T0, typename T1, std::size_t... Is>
        inline constexpr auto repeat_indexed(F&& f, T0&& t0, T1&& t1, std::index_sequence<Is...>) -> void
        {
            noop_t{((std::invoke(std::forward<F>(f), Is, meta::get<Is>(std::forward<T0>(t0)),
                                 meta::get<Is>(std::forward<T1>(t1)))),
                    0)...};
        }

        template<typename F, typename T0, std::size_t... Is>
        inline constexpr auto repeat_indexed(F&& f, T0&& t0, std::index_sequence<Is...>) -> void
        {
            noop_t{((std::invoke(std::forward<F>(f), Is, meta::get<Is>(std::forward<T0>(t0)))), 0)...};
        }

        template<typename T, typename F, std::size_t... Is>
        inline constexpr auto apply(T&& t, F&& f, std::index_sequence<Is...>)
        {
            return std::make_tuple(std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<T>(t)))...);
        }
    }   // namespace details

    template<typename T, typename U>
    inline constexpr auto tuple_sum_update(T&& lhs, U&& rhs) -> void
    {
        return details::tuple_sum_update(std::forward<T>(lhs), std::forward<U>(rhs),
                                         std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>{});
    }
    /**
     * @brief perform lhs += rhs on all the element of the tuple
     *
     * @tparam ArrayOfIt
     * @tparam TupleLike
     * @param lhs tuple or array of iterators
     * @param rhs the contribution tuple to add
     */
    template<typename ArrayOfIt, typename TupleLike>
    inline constexpr auto it_sum_update(ArrayOfIt&& lhs, TupleLike&& rhs) -> void
    {
        return details::it_sum_update(std::forward<ArrayOfIt>(lhs), std::forward<TupleLike>(rhs),
                                      std::make_index_sequence<meta::tuple_size<std::decay_t<ArrayOfIt>>::value>{});
    }

    template<typename T, typename U>
    inline constexpr auto tuple_min_update(T&& lhs, U&& rhs) -> void
    {
        return details::tuple_min_update(std::forward<T>(lhs), std::forward<U>(rhs),
                                         std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>{});
    }
    /**
     * @brief Perform  lhs = f(rhs1, rhs2) on each component of the three objects
     *
     * @param lhs  the left hand-side
     * @param rhs1  the first object on the right hand-side
     * @param rhs2 the the second object on the right hand-side
     * @param f    the lambda function with 2 arguments
     */
    template<typename F, typename LHS, typename RHS1, typename RHS2>
    inline constexpr auto for_each(LHS&& lhs, RHS1&& rhs1, RHS2&& rhs2, F&& f) -> void
    {
        return details::for_each(std::make_index_sequence<meta::tuple_size<std::decay_t<LHS>>::value>{},
                                 std::forward<F>(f), std::forward<LHS>(lhs), std::forward<RHS1>(rhs1),
                                 std::forward<RHS2>(rhs2));
    }
    /**
     * @brief Perform  lhs = f(rhs) on each component of the three objects
     *
     * @param lhs  the left hand-side
     * @param rhs  the right hand-side
     * @param f    the lambda function with one argument
     */
    template<typename T, typename U, typename F>
    inline constexpr auto for_each(T&& lhs, U&& rhs, F&& f) -> void
    {
        return details::for_each(std::forward<T>(lhs), std::forward<U>(rhs), std::forward<F>(f),
                                 std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>{});
    }
    /**
     * @brief Apply f on the tuple
     * \todo A reécrire

       * @param tuple
     * @param f the lambda to apply
     *    * Example
     *  \code
     *       meta::for_each(tuples, [&os](auto const& v) { os << v << ", "; });
     *       meta::for_each(tuples, [](auto const& v) { return 0; });
     * \endcode

     */
    template<typename T, typename F>
    inline constexpr auto for_each(T&& tuple, F&& f) -> void
    {
        return details::for_each(std::forward<T>(tuple), std::forward<F>(f),
                                 std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>{});
    }
    /**
     * @brief apply f on each component of t0,t1,t2,t3 (see the case of 2 objects)
     *
     * @param f  the lambda with 4 parameters
     * @param t0  first object
     * @param t1  second object
     * @param t2  third object
     * @param t3 fourth object
     */
    template<typename F, typename T0, typename T1, typename T2, typename T3>
    inline constexpr auto repeat(F&& f, T0&& t0, T1&& t1, T2&& t2, T3&& t3) -> void
    {
        details::repeat(std::forward<F>(f), std::forward<T0>(t0), std::forward<T1>(t1), std::forward<T2>(t2),
                        std::forward<T3>(t3), std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }
    /**
     * @brief apply f on each component of t0,t1,t2 (see the case of 2 objects)
     *
     * @param t1  second object
     * @param t2  third object
     */
    template<typename F, typename T0, typename T1, typename T2>
    inline constexpr auto repeat(F&& f, T0&& t0, T1&& t1, T2&& t2) -> void
    {
        details::repeat(std::forward<F>(f), std::forward<T0>(t0), std::forward<T1>(t1), std::forward<T2>(t2),
                        std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }
    /**
     * @brief  apply the function f on each component of t0, t1
     *
     * @param f   the lambda function with 2 arguments
     * @param t0  the first object
     * @param t1  the second object
     * example: perform contribution_force *= jacobian on each component of contribution_force and jacobian
     * \code meta::repeat([](auto& f, auto const& j) { f *= j; }, contribution_force, jacobian);
     * \endcode
     */
    template<typename F, typename T0, typename T1>
    inline constexpr auto repeat(F&& f, T0&& t0, T1&& t1) -> void
    {
        details::repeat(std::forward<F>(f), std::forward<T0>(t0), std::forward<T1>(t1),
                        std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }
    /**
     * @brief Repeat f on each element of t0
     *
     * @param f the lambda function to apply
     * @param t0 a set of elements (tuple, array)
     *
     * example:
     * In this example locals_iterator is an iterator tuple, the lambda function increments a value. The meta function
     * will increment each element of the tuple
     *  \code meta::repeat([](auto& it) { ++it; }, locals_iterator); \endcode
     */
    template<typename F, typename T0>
    inline constexpr auto repeat(F&& f, T0&& t0) -> void
    {
        details::repeat(std::forward<F>(f), std::forward<T0>(t0),
                        std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }

    template<typename F, typename T0, typename T1, typename T2, typename T3>
    inline constexpr auto repeat_indexed(F&& f, T0&& t0, T1&& t1, T2&& t2, T3&& t3) -> void
    {
        details::repeat_indexed(std::forward<F>(f), std::forward<T0>(t0), std::forward<T1>(t1), std::forward<T2>(t2),
                                std::forward<T3>(t3),
                                std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }

    template<typename F, typename T0, typename T1, typename T2>
    inline constexpr auto repeat_indexed(F&& f, T0&& t0, T1&& t1, T2&& t2) -> void
    {
        details::repeat_indexed(std::forward<F>(f), std::forward<T0>(t0), std::forward<T1>(t1), std::forward<T2>(t2),
                                std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }
    template<typename F, typename T0, typename T1>
    inline constexpr auto repeat_indexed(F&& f, T0&& t0, T1&& t1) -> void
    {
        details::repeat_indexed(std::forward<F>(f), std::forward<T0>(t0), std::forward<T1>(t1),
                                std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }

    template<typename F, typename T0>
    inline constexpr auto repeat_indexed(F&& f, T0&& t0) -> void
    {
        details::repeat_indexed(std::forward<F>(f), std::forward<T0>(t0),
                                std::make_index_sequence<meta::tuple_size<std::decay_t<T0>>::value>{});
    }

    template<typename T, typename F>
    inline constexpr auto apply(T&& t, F&& f)
    {
        return details::apply(std::forward<T>(t), std::forward<F>(f),
                              std::make_index_sequence<meta::tuple_size<std::decay_t<T>>::value>{});
    }

    template<std::size_t N>
    struct looper
    {
        template<typename F, std::size_t Dimension, typename... Is>
        constexpr inline auto operator()(F&& f, std::array<std::size_t, Dimension> const& stops, Is&... is)
        {
            for(std::size_t i = 0; i < meta::get<N - 1>(stops); ++i)
            {
                looper<N - 1>()(std::forward<F>(f), stops, is..., i);
            }
        }
    };

    template<>
    struct looper<1>
    {
        template<typename F, std::size_t Dimension, typename... Is>
        constexpr inline auto operator()(F&& f, std::array<std::size_t, Dimension> const& stops, Is&... is)
        {
            for(std::size_t i = 0; i < meta::get<0>(stops); ++i)
            {
                std::invoke(std::forward<F>(f), is..., i);
            }
        }
    };

    template<std::size_t N>
    struct looper_range
    {
        template<typename F, std::size_t Dimension, typename... Is>
        constexpr inline auto operator()(F&& f, std::array<int, Dimension> const& starts,
                                         std::array<int, Dimension> const& stops, Is&... is)
        {
            for(int i = meta::get<N - 1>(starts); i < meta::get<N - 1>(stops); ++i)
            {
                looper_range<N - 1>()(std::forward<F>(f), starts, stops, is..., i);
            }
        }
    };

    template<>
    struct looper_range<1>
    {
        template<typename F, std::size_t Dimension, typename... Is>
        constexpr inline auto operator()(F&& f, std::array<int, Dimension> const& starts,
                                         std::array<int, Dimension> const& stops, Is&... is)
        {
            for(int i = meta::get<0>(starts); i < meta::get<0>(stops); ++i)
            {
                std::invoke(std::forward<F>(f), is..., i);
            }
        }
    };

    template<typename ParticleOrTuple>
    auto inline as_tuple(ParticleOrTuple&& p)
    {
        if constexpr(is_particle_v<std::decay_t<ParticleOrTuple>>)
        {
            return std::forward<ParticleOrTuple>(p).as_tuple();
        }
        else if constexpr(is_tuple_v<std::decay_t<ParticleOrTuple>>)
        {
            return std::forward<ParticleOrTuple>(p);
        }
    }

    template<typename T>
    struct add_lvalue_reference : std::add_lvalue_reference<T>
    {
    };

    template<typename... Ts>
    struct add_lvalue_reference<std::tuple<Ts...>>
    {
        using type = std::tuple<Ts&...>;
    };

    template<typename... Ts>
    using add_lvalue_reference_t = typename add_lvalue_reference<Ts...>::type;

    template<typename T>
    struct add_const : std::add_const<T>
    {
    };

    template<typename... Ts>
    struct add_const<std::tuple<Ts...>>
    {
        using type = std::tuple<const Ts...>;
    };

    template<typename... Ts>
    using add_const_t = typename add_const<Ts...>::type;
    /**
     * @brief A constexpr pow function on integer
     *
     * The function is necessary for LLVM to compute \f N^D\ f
     *  \code {.c++}
     *    constexpr int nb = meta::Pow<3, 3>::value
     * \endcode
     *
     * @tparam N
     * @tparam D
     */
    template<int N, int D>
    struct Pow
    {
        enum
        {
            value = N * Pow<N, D - 1>::value
        };
    };

    template<int N>
    struct Pow<N, 0>
    {
        enum
        {
            value = 1
        };
    };
    struct empty_t
    {
        empty_t() = default;
        template<typename T, typename V>
        empty_t(T, V)
        {
        }
    };
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

}   // namespace scalfmm::meta
#endif   // SCALFMM_META_UTILS_HPP
