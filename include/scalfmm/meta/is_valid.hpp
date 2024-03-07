//==============================================================================
// This file is a test for a clean scalfmm code.
// It's a attempt to a cleaner and more expressive code for the user.
// Author : Pierre Esterie
//==============================================================================
#ifndef SCALFMM_META_IS_VALID_HPP
#define SCALFMM_META_IS_VALID_HPP

#include <type_traits>

namespace scalfmm::meta
{
    namespace details
    {
        template<typename F, typename... Args, typename = decltype(std::declval<F&&>()(std::declval<Args&&>()...))>
        constexpr auto is_valid_impl(int)
        {
            return std::true_type{};
        }

        template<typename F, typename... Args>
        constexpr auto is_valid_impl(...)
        {
            return std::false_type{};
        }

        template<typename F>
        struct is_valid_fun
        {
            template<typename... Args>
            constexpr auto operator()(Args&&...) const
            {
                return is_valid_impl<F, Args&&...>(int{});
            }
        };
    }   // namespace details

    struct is_valid_t
    {
        template<typename F>
        constexpr auto operator()(F&&) const
        {
            return details::is_valid_fun<F&&>{};
        }

        template<typename F, typename... Args>
        constexpr auto operator()(F&& f, Args&&... args) const
        {
            return details::is_valid_impl<F&&, Args&&...>(int{});
        }
    };

    template<typename T>
    struct type_w
    {
        using type = T;
    };

    template<typename T>
    constexpr type_w<T> type_c{};

    constexpr is_valid_t is_valid{};

    template<typename F, typename... Args>
    inline constexpr bool is_valid_v = [](F&& f, Args&&... args) constexpr -> bool
    {
        return decltype(is_valid(f, args...))::value;
    }();



}   // namespace scalfmm::meta

#endif
