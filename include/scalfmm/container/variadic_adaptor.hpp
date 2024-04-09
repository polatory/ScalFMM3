// --------------------------------
// See LICENCE file at project root
// File : container/variadic_adaptor.hpp
// --------------------------------
#ifndef SCALFMM_VARIADIC_ADAPTOR_HPP
#define SCALFMM_VARIADIC_ADAPTOR_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <scalfmm/container/reference_sequence.hpp>
#include <scalfmm/meta/is_valid.hpp>
#include <scalfmm/meta/traits.hpp>
#include <scalfmm/meta/utils.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <xsimd/config/xsimd_config.hpp>
#include <xtensor/xexpression.hpp>

namespace scalfmm
{
    namespace container
    {
        template<typename Derived, typename... Containers>
        struct variadic_adaptor;
    }   // namespace container
}   // namespace scalfmm

namespace scalfmm::container
{
    /// @brief
    ///
    /// @tparam Container
    /// @param C
    /// @param i
    ///
    /// @return
    template<typename Container>
    [[nodiscard]] inline constexpr auto id(Container C, std::size_t i)
    {
        return Container{};
    }

    /// @brief
    ///
    /// @tparam Containers
    /// @param Cs
    ///
    /// @return
    template<typename... Containers>
    [[nodiscard]] inline constexpr auto get_variadic_adaptor(Containers... Cs)
    {
        return variadic_adaptor<void, Containers...>{};
    }

    /// @brief
    ///
    /// @tparam Container
    /// @tparam Is
    /// @param s
    ///
    /// @return
    template<typename Container, std::size_t... Is>
    [[nodiscard]] inline constexpr auto get_variadic_adaptor(std::index_sequence<Is...> s)
    {
        return get_variadic_adaptor(id(Container{}, Is)...);
    }

    /// @brief
    ///
    /// @tparam Container
    /// @tparam Size
    ///
    /// @return
    template<typename Container, std::size_t Size>
    [[nodiscard]] inline constexpr auto get_variadic_adaptor()
    {
        return get_variadic_adaptor<Container>(std::make_index_sequence<Size>{});
    }

    /// @brief
    ///
    /// @tparam Container
    /// @tparam Size
    /// @param
    ///
    /// @return
    template<typename Container, std::size_t Size>
    using get_variadic_adaptor_t = decltype(get_variadic_adaptor<Container, Size>());

    template<typename Derived, typename... Containers>
    struct variadic_adaptor;

    /// @brief
    ///
    /// @tparam VariadicAdaptor
    /// @tparam DerivedVariadic
    /// @tparam Seq
    /// @tparam IsConst
    template<class VariadicAdaptor, typename DerivedVariadic, typename Seq, bool IsConst>
    class proxy_iterator
    {
        using container_private_type =
          std::conditional_t<std::is_base_of_v<VariadicAdaptor, DerivedVariadic>, VariadicAdaptor, DerivedVariadic>;
        using vector_pointer_type = std::conditional_t<IsConst, container_private_type const*, container_private_type*>;
        using derived_value_type = typename DerivedVariadic::value_type;

        friend container_private_type;
        vector_pointer_type vec_;
        int index_;

        proxy_iterator(vector_pointer_type vec, int index) noexcept
          : vec_{vec}
          , index_{index}
        {
        }

      public:
        // Need to adapt iterator category !!!
        using iterator_category = std::random_access_iterator_tag;
        using container_type = VariadicAdaptor;
        using tuple_of_ref =
          std::conditional_t<IsConst, typename VariadicAdaptor::const_reference, typename VariadicAdaptor::reference>;
        using value_type = typename VariadicAdaptor::value_type;
        using reference = tuple_of_ref;
        using pointer = void;
        using difference_type = std::size_t;
        static constexpr bool is_const_qualified{IsConst};

      private:
        template<size_t... Is>
        [[nodiscard]] [[nodiscard]] auto make_proxy(std::index_sequence<Is...> /*unused*/) const noexcept
        {
            // TODO : test with at() !!!
            return std::forward_as_tuple(std::get<Is>(*vec_).data()[index_]...);
        }

      public:
        proxy_iterator() = default;
        proxy_iterator(proxy_iterator const&) = default;
        proxy_iterator(proxy_iterator&&) = default;
        auto operator=(proxy_iterator const&) -> proxy_iterator& = default;
        auto operator=(proxy_iterator&&) -> proxy_iterator& = default;

        [[nodiscard]] auto operator*() const noexcept { return make_proxy(Seq{}); }

        [[nodiscard]] auto operator==(proxy_iterator const& rhs) const noexcept -> bool { return index_ == rhs.index_; }
        [[nodiscard]] auto operator!=(proxy_iterator const& rhs) const noexcept -> bool { return !(*this == rhs); }

        [[nodiscard]] auto operator<(proxy_iterator const& rhs) const noexcept -> bool { return index_ < rhs.index_; }
        [[nodiscard]] auto operator>(proxy_iterator const& rhs) const noexcept -> bool { return rhs < *this; }
        [[nodiscard]] auto operator<=(proxy_iterator const& rhs) const noexcept -> bool { return !(rhs < *this); }
        [[nodiscard]] auto operator>=(proxy_iterator const& rhs) const noexcept -> bool { return !(*this < rhs); }

        auto operator++() noexcept -> proxy_iterator& { return ++index_, *this; }
        auto operator--() noexcept -> proxy_iterator& { return --index_, *this; }
        auto operator++(int) noexcept -> proxy_iterator
        {
            const auto old = *this;
            return ++index_, old;
        }
        auto operator--(int) noexcept -> proxy_iterator
        {
            const auto old = *this;
            return --index_, old;
        }

        auto operator+=(int shift) noexcept -> proxy_iterator& { return index_ += shift, *this; }
        auto operator-=(int shift) noexcept -> proxy_iterator& { return index_ -= shift, *this; }

        auto operator+(int shift) const noexcept -> proxy_iterator { return {vec_, index_ + shift}; }
        auto operator-(int shift) const noexcept -> proxy_iterator { return {vec_, index_ - shift}; }

        auto operator-(proxy_iterator const& rhs) const noexcept -> int { return index_ - rhs.index_; }
    };

    /// @brief
    ///
    /// @tparam Derived
    /// @tparam Containers
    template<typename Derived, typename... Containers>
    struct variadic_adaptor : public std::tuple<Containers...>
    {
      public:
        using base_type = std::tuple<Containers...>;
        using self_type = variadic_adaptor<Derived, Containers...>;
        using size_type = std::tuple<typename Containers::size_type...>;
        using value_type = std::tuple<typename Containers::value_type...>;
        using derived_type = Derived;
        using const_reference = std::tuple<typename Containers::const_reference...>;
        using reference = std::tuple<typename Containers::reference...>;

        using allocator_type = std::tuple<typename Containers::allocator_type...>;
        using sequence_type = std::index_sequence_for<Containers...>;
        static constexpr auto indices = std::index_sequence_for<Containers...>{};
        using proxy_derived_type = std::conditional_t<std::is_void_v<derived_type>, self_type, derived_type>;
        using iterator =
          proxy_iterator<self_type, proxy_derived_type, std::make_index_sequence<sizeof...(Containers)>, false>;
        using const_iterator =
          proxy_iterator<self_type, proxy_derived_type, std::make_index_sequence<sizeof...(Containers)>, true>;

      private:
        friend iterator;
        friend const_iterator;

      public:
        variadic_adaptor() = default;
        variadic_adaptor(variadic_adaptor const&) = default;
        variadic_adaptor(variadic_adaptor&&) noexcept = default;
        auto operator=(variadic_adaptor const&) -> variadic_adaptor& = default;
        auto operator=(variadic_adaptor&&) noexcept -> variadic_adaptor& = default;
        ~variadic_adaptor() = default;
        // ===========================================================
        // Constructors
        // ===========================================================
        template<typename... Allocators, typename = std::enable_if_t<meta::all(
                                           std::is_same_v<typename Containers::allocator_type, Allocators>...)>>
        explicit variadic_adaptor(Allocators const&... alloc)
          : base_type(alloc...)
        {
        }

        //explicit variadic_adaptor(size_type counts, value_type const& values,
        //                          allocator_type const& allocators = allocator_type())
        //  //: variadic_adaptor(counts, values, allocators)//, std::index_sequence_for<Containers...>{})
        //  : base_type(Containers(typename Containers::size_type(count), value, allocator)...)
        //{
        //}

        template<
          typename T, typename Allocator = std::allocator<T>,
          typename = std::enable_if_t<meta::all((std::is_same_v<T, typename Containers::value_type> &&
                                                 std::is_same_v<Allocator, typename Containers::allocator_type>)...)>>
        explicit variadic_adaptor(std::size_t count, T const& value, Allocator const& allocator = Allocator())
          : base_type(Containers(typename Containers::size_type(count), value, allocator)...)
        {
        }

        explicit variadic_adaptor(size_type counts, allocator_type const& allocators = allocator_type())
          : variadic_adaptor(counts, allocators, std::index_sequence_for<Containers...>{})
        {
        }

        template<
          typename T, typename Allocator = std::allocator<T>,
          typename = std::enable_if_t<meta::all(std::is_same_v<Allocator, typename Containers::allocator_type>...)>>
        explicit variadic_adaptor(std::size_t count, Allocator const& allocator = Allocator())
          : base_type(Containers(typename Containers::size_type(count), allocator)...)
        {
        }

        explicit variadic_adaptor(size_type counts)
          : variadic_adaptor(counts, std::index_sequence_for<Containers...>{})
        {
        }

        explicit variadic_adaptor(std::size_t count)
          : base_type(Containers(typename Containers::size_type(count))...)
        {
        }

        // template<typename... Containers>
        explicit variadic_adaptor(Containers&&... cs)
          : base_type(std::forward<Containers>(cs)...)
        {
        }

        template<typename... Expressions,
                 typename = std::enable_if_t<meta::all(xt::is_xexpression<std::decay_t<Expressions>>::value...)>>
        explicit variadic_adaptor(Expressions&&... Es)
          : base_type(std::forward<Expressions>(Es)...)
        {
        }

        template<typename... Expressions,
                 typename = std::enable_if_t<meta::all(xt::is_xexpression<std::decay_t<Expressions>>::value...)>>
        explicit variadic_adaptor(std::tuple<Expressions...>&& tp_expr)
          : variadic_adaptor(tp_expr, std::index_sequence_for<Expressions...>{})
        {
        }

        // ===========================================================
        // Capacity
        // ===========================================================
        template<typename DelayedReturnType = std::tuple<decltype(meta::delayed_trait<Containers, meta::has_size_func>(
                   meta::has_size_func_f(Containers{})))...>>
        [[nodiscard]] constexpr auto all_size() const noexcept
          -> std::enable_if_t<meta::all(meta::has_size_func_v<Containers>...), DelayedReturnType>
        {
            std::tuple<meta::has_size_func_t<Containers>...> sizes;
            meta::for_each(sizes, *this, [](auto& container) { return container.size(); });
            return sizes;
        }

        template<typename DelayedReturnType = std::tuple<decltype(meta::delayed_trait<Containers, meta::has_empty_func>(
                   meta::has_empty_func_f(Containers{})))...>>
        [[nodiscard]] constexpr auto empty() const noexcept
          -> std::enable_if_t<meta::all(meta::has_empty_func_v<Containers>...), DelayedReturnType>
        {
            std::tuple<meta::has_empty_func_t<Containers>...> bools;
            meta::for_each(bools, *this, [](auto& container) { return container.empty(); });
            return bools;
        }

        template<
          typename DelayedReturnType = std::tuple<decltype(meta::delayed_trait<Containers, meta::has_max_size_func>(
            meta::has_max_size_func_f(Containers{})))...>>
        [[nodiscard]] constexpr auto max_size() const noexcept
          -> std::enable_if_t<meta::all(meta::has_max_size_func_v<Containers>...), DelayedReturnType>
        {
            std::tuple<meta::has_max_size_func_t<Containers>...> bools;
            meta::for_each(bools, *this, [](auto& container) { return container.max_size(); });
            return bools;
        }

        template<typename DelayedReturnType = void>
        [[nodiscard]] constexpr auto resize(std::size_t count)
          -> std::enable_if_t<meta::all(meta::has_resize_func_v<Containers>...), DelayedReturnType>
        {
            meta::repeat([count](auto& container) { container.resize(count); }, *this);
        }

        template<typename T, typename DelayedReturnType = void>
        constexpr auto resize(std::size_t count, T const& value)
          -> std::enable_if_t<meta::all(meta::has_resize_valued_func_v<Containers> &&
                                          std::is_same<T, typename Containers::value_type>::value...),
                              DelayedReturnType>
        {
            meta::repeat([count, &value](auto& container) { container.resize(count, value); }, *this);
        }

        template<typename DelayedReturnType = void>
        constexpr auto clear() -> std::enable_if_t<meta::all(meta::has_clear_func_v<Containers>...), DelayedReturnType>
        {
            meta::repeat([](auto& container) { container.clear(); }, *this);
        }

        // ===========================================================
        // Iterators
        // ===========================================================
      public:
        [[nodiscard]] inline auto begin() -> std::enable_if_t<meta::all(meta::has_begin_v<Containers>...), iterator>
        {
            return {this, 0};
        }

        [[nodiscard]] inline auto begin() const
          -> std::enable_if_t<meta::all(meta::has_cbegin_v<Containers>...), const_iterator>
        {
            return {this, 0};
        }

        [[nodiscard]] inline auto cbegin() const
          -> std::enable_if_t<meta::all(meta::has_cbegin_v<Containers>...), const_iterator>
        {
            return {this, 0};
        }

        [[nodiscard]] inline auto end() -> std::enable_if_t<meta::all(meta::has_end_v<Containers>...), iterator>
        {
            return {this, static_cast<int>(std::get<0>(*this).size())};
        }

        [[nodiscard]] inline auto end() const
          -> std::enable_if_t<meta::all(meta::has_cend_v<Containers>...), const_iterator>
        {
            return {this, static_cast<int>(std::get<0>(*this).size())};
        }

        [[nodiscard]] inline auto cend() const
          -> std::enable_if_t<meta::all(meta::has_cend_v<Containers>...), const_iterator>
        {
            return {this, static_cast<int>(std::get<0>(*this).size())};
        }

        template<typename Seq>
        [[nodiscard]] inline auto sbegin() -> std::enable_if_t<meta::all(meta::has_begin_v<Containers>...),
                                                               proxy_iterator<self_type, derived_type, Seq, false>>
        {
            return {this, 0};
        }

        template<typename Seq>
        [[nodiscard]] inline auto sbegin() const -> std::enable_if_t<meta::all(meta::has_begin_v<Containers>...),
                                                                     proxy_iterator<self_type, derived_type, Seq, true>>
        {
            return {this, 0};
        }

        template<typename Seq>
        [[nodiscard]] inline auto send() -> std::enable_if_t<meta::all(meta::has_end_v<Containers>...),
                                                             proxy_iterator<self_type, derived_type, Seq, false>>
        {
            return {this, static_cast<int>(std::get<0>(*this).size())};
        }

        template<typename Seq>
        [[nodiscard]] inline auto send() const -> std::enable_if_t<meta::all(meta::has_end_v<Containers>...),
                                                                   proxy_iterator<self_type, derived_type, Seq, true>>
        {
            return {this, static_cast<int>(std::get<0>(*this).size())};
        }
        // at, WARNING changing of behavior, here at was return the vector in the sequence
        // now it returns the refs from the tuple sequence like the subscript
        // operator. TODO: this should throw to be standart compliant.
        [[nodiscard]] inline auto at(std::size_t i)
        {
            auto it = this->begin()+i;
            return *it;
        }
        [[nodiscard]] inline auto at(std::size_t i) const
        {
            auto it = this->cbegin()+i;
            return *it;
        }

        [[nodiscard]] inline auto operator[](std::size_t i)
        {
            auto it = this->begin()+i;
            return *it;
        }
        [[nodiscard]] inline auto operator[](std::size_t i) const
        {
            auto it = this->cbegin()+i;
            return *it;
        }
    };

    // Use the same container in a variadic_adaptor

    /// @brief
    ///
    /// @tparam Derived
    /// @tparam U
    /// @tparam Allocator
    template<typename Derived, template<typename U, typename Allocator = XTENSOR_DEFAULT_ALLOCATOR(U)> class Container,
             typename... Types>
    struct unique_variadic_container : public variadic_adaptor<Derived, Container<Types>...>
    {
        using base_type = variadic_adaptor<Derived, Container<Types>...>;
        using base_type::base_type;
    };

    /// @brief
    ///
    /// @tparam Derived
    /// @tparam Types
    template<typename Derived, typename... Types>
    struct variadic_container : public unique_variadic_container<Derived, std::vector, Types...>
    {
        using base_type = unique_variadic_container<Derived, std::vector, Types...>;
        using base_type::base_type;
    };

    /// @brief
    ///
    /// @tparam Derived
    /// @tparam Tuple
    template<typename Derived, typename Tuple>
    struct variadic_container_tuple;

    template<typename Derived, typename... Types>
    struct variadic_container_tuple<Derived, std::tuple<Types...>> : public variadic_container<Derived, Types...>
    {
        using base_type = variadic_container<Derived, Types...>;
        using base_type::base_type;
    };

}   // namespace scalfmm::container
#endif   // SCALFMM_VARIADIC_ADAPTOR_HPP
