// --------------------------------
// See LICENCE file at project root
// File : kernels/kernels.hpp
// --------------------------------
#ifndef SCALFMM_MATRIX_KERNELS_SCALAR_KERNELS_HPP
#define SCALFMM_MATRIX_KERNELS_SCALAR_KERNELS_HPP

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
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>

#include "scalfmm/meta/utils.hpp"

namespace scalfmm::matrix_kernels::others
{
    ///////
    /// \brief The one_over_r2 struct
    ///
    ///   The kernel \f$k(x,y) : R^{km} -> R^{kn}\f$  with \f$ km = kn =1\f$ <p>
    ///          \f$  k(x,y) = | x - y |^{-2} \f$  with  \f$x \in  R^d \f$
    ///
    ///  For  \f$d = 2 \f$ <p>
    ///   \f$x = (x1,x2), y = (y1,y2)\f$  and  \f$|x-y|^2 = (x1-y1)^2 +  (x2-y2)^2 \f$
    ///
    /// The kernel is homogeneous  \f$k(ax,ay) = 1/a^2 k(x,y) \f$
    ///
    /// scale factor  is  \f$1/a^2 \f$
    ///
    struct one_over_r2
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{1};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("one_over_r2"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>{ValueType(1.)};
        }

        template<typename ValueType1, typename ValueType2, std::size_t Dim>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, Dim> const& x,
                                           container::point<ValueType2, Dim> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dim>{});
        }

        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            return vector_type<ValueType>({ValueType(1.) / (cell_width * cell_width)});
        }

        template<typename ValueType1, typename ValueType2, std::size_t Dim, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, Dim> const& xs,
                                                    container::point<ValueType2, Dim> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
        {
            using decayed_type = typename std::decay_t<ValueType1>;

            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             (((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...)};
        }
        static constexpr int separation_criterion{1};
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief grad_one_over_r matrix kernel to compute the gradient of Laplace kernel.
    ///
    /// grad_one_over_r\f$ k(x,y) : R -> R^{d}\f$ with \f$d \f$ the space
    /// dimension.
    ///
    ///           \f$k(x,y) = \grad | x - y |^{-2}\f$
    ///  For  \f$d = 2 \f$ <p>
    ///   \f$x = (x1,x2), y = (y1,y2)\f$  and  \f$|x-y|^2 = (x1-y1)^2 +  (x2-y2)^2 \f
    ///
    ///   \f$k(x,y) = \grad | x - y |^{-2} = \frac{-2 *(x-y)}{|x-y|^4 }\f$
    ///
    /// The kernel is homogeneous  \f$k(ax,ay) = 1/a^3 k(x,y) \f$
    ///
    /// scale factor  is  \f$1/a^3 \f$
    ///
    template<std::size_t Dim = 3>
    struct grad_one_over_r2
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{Dim};
        const std::string name() const { return std::string("grad_one_over_r2<" + std::to_string(Dim) + ">"); }

        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            vector_type<ValueType> mc;
            std::fill(std::begin(mc), std::end(mc), ValueType(-1.));
            return mc;
        }

        template<typename ValueType>
        [[nodiscard]] inline auto evaluate(container::point<ValueType, Dim> const& x,
                                           container::point<ValueType, Dim> const& y) const noexcept
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dim>{});
        }

        //
        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            auto tmp{math::pow(ValueType(1.) / cell_width, 3)};
            vector_type<ValueType> sf;
            std::fill(std::begin(sf), std::end(sf), tmp);
            return sf;
        }

        template<typename ValueType, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType, Dim> const& xs,
                                                    container::point<ValueType, Dim> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
        {
            using std::sqrt;
            using decayed_type = typename std::decay_t<ValueType>;

            ValueType tmp = decayed_type(1.0) / (((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...);
            ValueType r4{pow(tmp, 2)};
            return matrix_type<decayed_type>{(ValueType(-2.0) * r4 * (xs.at(Is) - ys.at(Is)))...};
            //-2.0*r4 * (xs - ys);
        }

        static constexpr int separation_criterion{1};
    };

}   // namespace scalfmm::matrix_kernels::others

#endif
