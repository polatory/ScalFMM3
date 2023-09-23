// --------------------------------
// See LICENCE file at project root
// File : matrix_kernels/debug.hpp
// --------------------------------
#ifndef SCALFMM_MATRIX_KERNELS_DEBUG_HPP
#define SCALFMM_MATRIX_KERNELS_DEBUG_HPP

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/mk_common.hpp>
#include <scalfmm/meta/utils.hpp>
#include <scalfmm/utils/math.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <xtensor/xmath.hpp>

#include "scalfmm/meta/utils.hpp"

namespace scalfmm::matrix_kernels::debug
{
    // This matrix kernel is here to debug the non homogenous case
    struct one_over_r_non_homogenous
    {
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};
        static constexpr auto symmetry_tag{symmetry::symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{1};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("one_over_r_non_homogenous"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>{ValueType(1.)};
        }

        template<typename ValueType, std::size_t Dim>
        [[nodiscard]] inline auto evaluate(container::point<ValueType, Dim> const& x,
                                           container::point<ValueType, Dim> const& y) const noexcept
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dim>{});
        }

        template<typename ValueType, std::size_t Dim, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType, Dim> const& xs,
                                                    container::point<ValueType, Dim> const& ys,
                                                    std::index_sequence<Is...> is) const noexcept
        {
            using std::sqrt;
            using decayed_type = typename std::decay_t<ValueType>;

            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
        }
        static constexpr int separation_criterion{1};
    };

    struct one_over_r_non_symmetric
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::non_symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{1};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("one_over_r_non_symmetric"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>{ValueType(1.)};
        }

        template<typename ValueType, std::size_t Dim>
        [[nodiscard]] inline auto evaluate(container::point<ValueType, Dim> const& x,
                                           container::point<ValueType, Dim> const& y) const noexcept
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dim>{});
        }

        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            return vector_type<ValueType>{ValueType(1.) / cell_width};
        }

        template<typename ValueType, std::size_t Dim, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType, Dim> const& xs,
                                                    container::point<ValueType, Dim> const& ys,
                                                    std::index_sequence<Is...> is) const noexcept
        {
            using std::sqrt;
            using decayed_type = typename std::decay_t<ValueType>;
            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
        }
        static constexpr int separation_criterion{1};
    };
}   // namespace scalfmm::matrix_kernels::debug
#endif   // SCALFMM_MATRIX_KERNELS_DEBUG_HPP
