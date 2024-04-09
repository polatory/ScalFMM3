// --------------------------------
// See LICENCE file at project root
// File : utils/math.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_MATH_HPP
#define SCALFMM_UTILS_MATH_HPP

#include <cfloat>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace scalfmm::math
{
    template<typename T>
    inline auto factorial(int value) -> T
    {
        if(value == 0)
        {
            return T{1};
        }

        int result = value;
        while(--value > 1)
        {
            result *= value;
        }
        return T(result);
    }

    template<typename T>
    inline auto pow(T a, int p) -> T
    {
        T result{T(1.)};
        while(p-- > 0)
        {
            result *= a;
        }
        return result;
    }

    template<typename T>
    inline constexpr auto pow(T a, std::size_t p) -> T
    {
        return p == 0 ? 1 : a * pow<T>(a, p - 1);
    }

    template<typename T, typename U, typename = std::enable_if_t<std::is_floating_point<T>::value, T>,
             typename = std::enable_if_t<std::is_floating_point<U>::value, U>>
    inline constexpr auto feq(const T& a, const U& b, T epsilon = std::numeric_limits<T>::epsilon()) -> bool
    {
        return a - b < epsilon && b - a < epsilon;
    }

    template<typename ValueType1, typename ValueType>
    inline constexpr auto between(ValueType1 value, ValueType range_begin, ValueType range_end) -> bool
    {
        return (value >= range_begin && value < range_end);
    }

}   // namespace scalfmm::math

#endif   // SCALFMM_UTILS_MATH_HPP
