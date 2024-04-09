//==============================================================================
// This file is a test for a clean scalfmm code.
// It's a attempt to a cleaner and more expressive code for the user.
// Author : Pierre Esterie
//==============================================================================
#ifndef SCALFMM_META_TRAITS_HPP
#define SCALFMM_META_TRAITS_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <scalfmm/meta/forward.hpp>
#include <scalfmm/meta/is_valid.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <xsimd/xsimd.hpp>

namespace scalfmm
{
    namespace container
    {
        template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
                 typename OutputsType, std::size_t MOutputs, typename... Variables>
        struct particle;
    }   // namespace container
}   // namespace scalfmm
namespace xsimd
{
    template<class T, class A>
    class batch;
}   // namespace xsimd

// Traits
namespace scalfmm::meta
{
    template<std::size_t Whatever, typename T>
    inline constexpr auto id(T&& t)
    {
        return std::forward<T>(t);
    }

    template<typename T, template<typename> class Trait>
    inline constexpr auto delayed_trait(std::true_type)
    {
        return typename Trait<T>::type{};
    }

    struct foo
    {
    };

    template<typename T, template<typename> class Trait>
    inline constexpr auto delayed_trait(std::false_type b)
    {
        return foo{};
    }

    inline constexpr auto is_equality_comparable_f = meta::is_valid([](auto&& a, auto&& b) -> decltype(a == b) {});
    inline constexpr auto has_begin_f = meta::is_valid([](auto&& a) -> decltype(a.begin()) {});
    inline constexpr auto has_cbegin_f = meta::is_valid([](auto&& a) -> decltype(a.cbegin()) {});
    inline constexpr auto has_rbegin_f = meta::is_valid([](auto&& a) -> decltype(a.rbegin()) {});
    inline constexpr auto has_crbegin_f = meta::is_valid([](auto&& a) -> decltype(a.crbegin()) {});
    inline constexpr auto has_end_f = meta::is_valid([](auto&& a) -> decltype(a.end()) {});
    inline constexpr auto has_cend_f = meta::is_valid([](auto&& a) -> decltype(a.cend()) {});
    inline constexpr auto has_rend_f = meta::is_valid([](auto&& a) -> decltype(a.rend()) {});
    inline constexpr auto has_crend_f = meta::is_valid([](auto&& a) -> decltype(a.crend()) {});
    inline constexpr auto has_plus_f = meta::is_valid([](auto&& a, auto&& b) -> decltype(a + b) {});
    inline constexpr auto has_size_func_f = meta::is_valid([](auto t) -> decltype(t.size()) {});
    inline constexpr auto has_max_size_func_f = meta::is_valid([](auto t) -> decltype(t.max_size()) {});
    inline constexpr auto has_empty_func_f = meta::is_valid([](auto t) -> decltype(t.empty()) {});
    inline constexpr auto has_resize_func_f = meta::is_valid([](auto t, auto count) -> decltype(t.resize(count)) {});
    inline constexpr auto has_resize_valued_func_f =
      meta::is_valid([](auto t, auto count, auto value) -> decltype(t.resize(count, value)) {});
    inline constexpr auto has_clear_func_f = meta::is_valid([](auto t) -> decltype(t.clear()) {});

    // Containers related traits
    template<class T>
    struct has_begin
    {
        using type = decltype(std::declval<std::decay_t<T>>().begin());
    };

    template<class T>
    struct has_cbegin
    {
        using type = decltype(std::declval<std::decay_t<T>>().cbegin());
    };

    template<class T>
    struct has_rbegin
    {
        using type = decltype(std::declval<std::decay_t<T>>().rbegin());
    };

    template<class T>
    struct has_crbegin
    {
        using type = decltype(std::declval<std::decay_t<T>>().crbegin());
    };

    template<typename T>
    struct has_end
    {
        using type = decltype(std::declval<std::decay_t<T>>().end());
    };

    template<typename T>
    struct has_cend
    {
        using type = decltype(std::declval<std::decay_t<T>>().cend());
    };

    template<typename T>
    struct has_rend
    {
        using type = decltype(std::declval<std::decay_t<T>>().rend());
    };

    template<typename T>
    struct has_crend
    {
        using type = decltype(std::declval<std::decay_t<T>>().crend());
    };

    template<class T>
    struct has_range_interface
    {
        static const bool value = has_begin<T>::value && has_end<T>::value;
    };

    template<typename T>
    struct has_size_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().size());
    };

    template<typename T>
    struct has_empty_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().empty());
    };

    template<typename T>
    struct has_max_size_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().max_size());
    };

    template<typename T>
    struct has_resize_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}));
    };

    template<typename T>
    struct has_resize_valued_func
    {
        using type =
          decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}, typename T::value_type{}));
    };

    template<typename T>
    struct has_clear_func
    {
        using type = decltype(std::declval<std::decay_t<T>>().clear());
    };

    // Type version
    template<typename T>
    using has_begin_t = decltype(std::declval<std::decay_t<T>>().begin());
    template<typename T>
    using has_cbegin_t = decltype(std::declval<std::decay_t<T>>().cbegin());
    template<typename T>
    using has_rbegin_t = decltype(std::declval<std::decay_t<T>>().rbegin());
    template<typename T>
    using has_crbegin_t = decltype(std::declval<std::decay_t<T>>().crbegin());
    template<typename T>
    using has_end_t = decltype(std::declval<std::decay_t<T>>().end());
    template<typename T>
    using has_cend_t = decltype(std::declval<std::decay_t<T>>().cend());
    template<typename T>
    using has_rend_t = decltype(std::declval<std::decay_t<T>>().rend());
    template<typename T>
    using has_crend_t = decltype(std::declval<std::decay_t<T>>().crend());
    template<typename T>
    using has_size_func_t = decltype(std::declval<std::decay_t<T>>().size());
    template<typename T>
    using has_empty_func_t = decltype(std::declval<std::decay_t<T>>().empty());
    template<typename T>
    using has_max_size_func_t = decltype(std::declval<std::decay_t<T>>().max_size());
    template<typename T>
    using has_resize_func_t = decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}));
    template<typename T>
    using has_resize_valued_func_t =
      decltype(std::declval<std::decay_t<T>>().resize(typename T::size_type{}, typename T::value_type{}));
    template<typename T>
    using has_clear_func_t = decltype(std::declval<std::decay_t<T>>().clear());

    // Value version
    template<typename T>
    inline constexpr bool has_begin_v = decltype(has_begin_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_cbegin_v = decltype(has_cbegin_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_rbegin_v = decltype(has_rbegin_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_crbegin_v = decltype(has_crbegin_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_end_v = decltype(has_end_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_cend_v = decltype(has_cend_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_rend_v = decltype(has_rend_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_crend_v = decltype(has_crend_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_range_interface_v = has_range_interface<T>::value;
    template<typename T>
    inline constexpr bool has_size_func_v = decltype(has_size_func_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_empty_func_v = decltype(has_empty_func_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_max_size_func_v = decltype(has_max_size_func_f(T{}))::value;
    template<typename T>
    inline constexpr bool has_resize_func_v = decltype(has_resize_func_f(T{}, typename T::size_type{}))::value;
    template<typename T>
    inline constexpr bool has_resize_valued_func_v =
      decltype(has_resize_valued_func_f(T{}, typename T::size_type{}, typename T::value_type{}))::value;
    template<typename T>
    inline constexpr bool has_clear_func_v = decltype(has_clear_func_f(T{}))::value;

    // Other traits

    template<class T>
    struct is_equality_comparable
    {
        static const bool value = decltype(is_equality_comparable_f(T{}, T{}))::value;
    };
    /**
     * @brief check if the type is float (std::complex) or not
     *
     * @tparam T
     */
    template<class, class = std::void_t<>>
    struct is_float : std::false_type
    {
    };

    // specialization recognizes types that do have a nested ::type member:
    template<class T>
    struct is_float<T, std::void_t<float>> : std::true_type
    {
    };

    /**
     * @brief check if the type is double (std::complex) or not
     *
     * @tparam T
     */
    template<class, class = std::void_t<>>
    struct is_double : std::false_type
    {
    };
    template<class T>
    struct is_double<T, std::void_t<double>> : std::true_type
    {
    };
    /**
     * @brief  check if the type is complex (std::complex) or not
     *
     * @tparam T
     */
    template<typename, typename = std::void_t<>>
    struct is_complex : std::false_type
    {
    };

    template<typename T>
    struct is_complex<std::complex<T>, std::void_t<std::complex<T>>> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_complex_v = is_complex<T>::value;

    //
    // Tuple cat
    template<typename T, typename U>
    struct tuple_cat
    {
        using type = decltype(std::tuple_cat(T{}, U{}));
    };

    /**
     * @brief Check if the type T has a member type value_type
     *
     * \code
     *
     * \endcode
     * @tparam T
     */
    template<typename T, typename = std::void_t<>>
    struct has_value_type : std::false_type
    {
        using type = T;
    };

    template<typename T>
    struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type
    {
        using type = typename T::value_type;
    };
    template<typename T>
    inline constexpr bool has_value_type_v = has_value_type<T>::value;

    template<typename T>
    using has_value_type_t = typename has_value_type<T>::type;

    template<typename T>
    struct is_addable
    {
        static const bool value = decltype(has_plus_f(T{}, T{}))::value;
    };

    template<typename T>
    struct is_simd : xsimd::is_batch<T>
    {
    };

    // tree related type traits
    template<typename, typename = std::void_t<>>
    struct is_leaf_group_symbolics : std::false_type
    {
    };

    template<typename T>
    struct is_leaf_group_symbolics<T, std::void_t<typename T::group_leaf_type>> : std::true_type
    {
    };

    template<typename T>
    struct is_arithmetic : std::is_arithmetic<T>
    {
    };
    template<typename T>
    struct is_integral : std::is_integral<T>
    {
    };

    template<typename T>
    struct is_arithmetic<xsimd::batch<T>> : is_arithmetic<T>
    {
    };

    template<typename T>
    struct is_integral<xsimd::batch<T>> : is_integral<T>
    {
    };

    template<typename T>
    struct is_tuple : std::false_type
    {
    };
    template<typename... Ts>
    struct is_tuple<std::tuple<Ts...>> : std::true_type
    {
    };
    template<typename... Ts>
    inline constexpr bool is_tuple_v = is_tuple<Ts...>::value;

    template<typename T>
    struct is_array : std::false_type
    {
    };
    template<typename T, std::size_t N>
    struct is_array<std::array<T, N>> : std::true_type
    {
    };
    template<typename... Ts>
    inline constexpr bool is_array_v = is_array<Ts...>::value;

    template<typename T>
    struct is_particle : std::false_type
    {
    };
    template<typename PositionType, std::size_t PositionDim, typename InputsType, std::size_t NInputs,
             typename OutputsType, std::size_t MOutputs, typename... Variables>
    struct is_particle<
      container::particle<PositionType, PositionDim, InputsType, NInputs, OutputsType, MOutputs, Variables...>>
      : std::true_type
    {
    };
    template<typename T>
    inline static constexpr bool is_particle_v = is_particle<T>::value;

    template<typename T>
    struct td;

    // interpolator
    inline constexpr auto sig_preprocess_f = meta::is_valid(
      [](auto&& inter, auto& cell, std::size_t th) -> decltype(inter.apply_multipoles_preprocessing_impl(cell, th)) {});
    inline constexpr auto sig_postprocess_f =
      meta::is_valid([](auto&& inter, auto& cell, auto& buffer,
                        std::size_t th) -> decltype(inter.apply_multipoles_postprocessing_impl(cell, buffer, th)) {});
    inline constexpr auto sig_buffer_init_f =
      meta::is_valid([](auto&& inter) -> decltype(inter.buffer_initialization_impl()) {});
    inline constexpr auto sig_buffer_shape_f =
      meta::is_valid([](auto&& inter) -> decltype(inter.buffer_shape_impl()) {});
    inline constexpr auto sig_buffer_reset_f =
      meta::is_valid([](auto&& inter, auto& buf) -> decltype(inter.buffer_reset_impl(buf)) {});
    inline constexpr auto sig_init_k_f = meta::is_valid([](auto&& inter) -> decltype(inter.initialize_k_impl()) {});
    inline constexpr auto sig_gen_k_f =
      meta::is_valid([](auto&& inter, auto&& X, auto&& Y, std::size_t n, std::size_t m,
                        std::size_t th) -> decltype(inter.generate_matrix_k_impl(X, Y, n, m, th)) {});
    inline constexpr auto sig_gen_w_f =
      meta::is_valid([](auto&& inter, std::size_t order) -> decltype(inter.generate_weights_impl(order)) {});
    /**
     * @brief Internal structure used for interaction list in source target simulation
     *
     * @tparam T
     */
    template<typename T>
    struct inject
    {
    };

    template<typename T, typename = void>
    struct exist : std::false_type
    {
        using type = void;
    };

    template<typename T>
    struct exist<T, std::void_t<typename T::type>> : std::true_type
    {
        using type = typename T::type;
    };

    template<typename T>
    static constexpr bool exist_v = exist<T>::value;
    template<typename T>
    using exist_t = typename exist<T>::type;

}   // namespace scalfmm::meta

#endif
