// --------------------------------
// See LICENCE file at project root
// File : kernels/kernels.hpp
// --------------------------------
#ifndef SCALFMM_MATRIX_KERNELS_LAPLACE_HPP
#define SCALFMM_MATRIX_KERNELS_LAPLACE_HPP

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "scalfmm/meta/utils.hpp"
#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/mk_common.hpp>
#include <scalfmm/meta/utils.hpp>
#include <scalfmm/utils/math.hpp>
#include <xtensor/xtensor_forward.hpp>

namespace scalfmm::matrix_kernels::laplace
{
    ///////
    /// \brief The one_over_r struct corresponds to the  \f$ 1/r\f$ kernel
    ///
    ///   The kernel \f$k(x,y): R^{km} -> R^{kn}\f$  with\f$ kn = km = 1\f$
    ///           \f$k(x,y) = | x - y |^{-1}\f$
    /// The kernel is homogeneous\f$ k(ax,ay) = 1/a k(x,y)\f$
    /// scale factor  is \f$1/a\f$
    ///
    struct one_over_r
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{1};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("one_over_r"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            using decayed_type = typename std::decay_t<ValueType>;
            return vector_type<decayed_type>{decayed_type(1.)};
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
            return vector_type<ValueType>{ValueType(1.) / cell_width};
        }

        template<typename ValueType, std::size_t Dim, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType, Dim> const& xs,
                                                    container::point<ValueType, Dim> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
        {
            using decayed_type = typename std::decay_t<ValueType>;
            return matrix_type<decayed_type>{decayed_type(1.0) /
                                             xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...))};
        }
        static constexpr int separation_criterion{1};
    };

    ///////
    /// \brief The like_mrhs simulates two FMMs with independent charges
    ///
    ///   The kernel \f$k(x,y): R^2 -> R^2\f$
    ///                    \f$ (q1,q2) --> (p1,p2)\f$
    ///           \f$k(x,y) = | x - y |^{-1} Id_{2x2}\f$
    /// The kernel is homogeneous \f$k(ax,ay) = 1/a k(x,y)\f$
    /// scale factor  is\f$ (1/a, 1/a)\f$
    struct like_mrhs
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::symmetric};
        static constexpr std::size_t km{2};
        static constexpr std::size_t kn{2};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>({ValueType(1.), ValueType(1.)});
        }

        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            auto tmp = ValueType(1.) / cell_width;
            return vector_type<ValueType>{tmp, tmp};
        }

        template<typename ValueType, std::size_t Dim>
        [[nodiscard]] inline auto evaluate(container::point<ValueType, Dim> const& x,
                                           container::point<ValueType, Dim> const& y) const noexcept
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dim>{});
        }
        ///
        /// @return the matrix K stored by rows in a tuple of size kn x mm organiz by rows
        ///
        template<typename ValueType1, typename ValueType2, std::size_t Dim, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, Dim> const& xs,
                                                    container::point<ValueType2, Dim> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
        {
            using decayed_type = typename std::decay_t<ValueType1>;

            auto val = decayed_type(1.0) / xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...));
            return matrix_type<decayed_type>{val, decayed_type(0.), decayed_type(0.), val};
        }

        static constexpr int separation_criterion{1};
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief grad_one_over_r matrix kernel to compute the gradient of Laplace kernel.
    ///
    /// grad_one_over_r\f$ k(x,y) : R -> R^{d}\f$ with \f$d \f$ the space
    /// dimension.
    ///
    ///           \f$k(x,y) = \grad | x - y |^{-1} = -(x-y) | x - y |^{-3}\f$
    ///
    /// scale factor  \f$k(ax,ay)= 1/a^2 k(x,y)\f$
    template<std::size_t Dim = 3>
    struct grad_one_over_r
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::non_symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{Dim};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("grad_one_over_r<" + std::to_string(Dim) + ">"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            vector_type<ValueType> mc;
            std::fill(std::begin(mc), std::end(mc), ValueType(-1.));
            return mc;
        }

        template<typename ValueType1, typename ValueType2>
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
            auto tmp{math::pow(ValueType(1.) / cell_width, 2)};
            vector_type<ValueType> sf;
            std::fill(std::begin(sf), std::end(sf), tmp);
            return sf;
        }

        template<typename ValueType1, typename ValueType2, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, Dim> const& xs,
                                                    container::point<ValueType2, Dim> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
        {
            using decayed_type = typename std::decay_t<ValueType1>;

            decayed_type tmp =
              decayed_type(1.0) / xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...));
            decayed_type r3{xsimd::pow(tmp, int(3))};
            return matrix_type<decayed_type>{r3 * (ys.at(Is) - xs.at(Is))...};
            //-r3 * (xs - ys);
        }

        static constexpr int separation_criterion{1};
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief The val_grad_one_over_r struct is the matrix kernel to compute the value and the gradient of the Laplace
    /// kernel.
    ///
    /// val_grad_one_over_r \f$k(x,y) : R^{km} -> R^{kn}\f$  with \f$ km = 1\f$ and \f$kn = d+1\f$  with \f$ d\f$ the
    /// space dimension.
    ///
    /// \f$k(x,y) = ( | x - y |^{-1}, \grad | x - y |^{-1} ) =(| x - y |^{-1}, -(x-y) | x - y |^{-3}\f)$
    ///
    ///  Is a specific kernel used to compute the  value of the kernel and its gradient
    ///
    /// scale factor \f$ k(ax,ay)= (1/a, 1/a^2 ... 1/a^2) k(x,y)\f$
    template<std::size_t Dim = 3>
    struct val_grad_one_over_r
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::non_symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{1 + Dim};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("val_grad_one_over_r<" + std::to_string(Dim) + ">"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            using decayed_type = typename std::decay_t<ValueType>;
            vector_type<decayed_type> mc;
            mc.fill(decayed_type(-1.));
            mc.at(0) = decayed_type(1.0);
            return mc;
        }

        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, Dim> const& x,
                                           container::point<ValueType2, Dim> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dim>{});
        }

        // No meaning
        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            using decayed_type = typename std::decay_t<ValueType>;
            auto tmp{decayed_type(1.) / cell_width};
            auto tmp_{math::pow(tmp, 2)};
            vector_type<decayed_type> sf;
            sf.fill(tmp_);
            sf.at(0) = tmp;
            return sf;
        }

        template<typename ValueType1, typename ValueType2, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, Dim> const& xs,
                                                    container::point<ValueType2, Dim> const& ys,
                                                    std::index_sequence<Is...>) const noexcept
        {
            using decayed_type = typename std::decay_t<ValueType1>;

            decayed_type tmp =
              decayed_type(1.0) / xsimd::sqrt((((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...));
            decayed_type r3{xsimd::pow(tmp, int(3))};
            return matrix_type<decayed_type>{tmp, (r3 * (ys.at(Is) - xs.at(Is)))...};   //-r3 * (xs - ys);
        }

        static constexpr int separation_criterion{1};
    };

    ///////
    /// \brief The ln_2d struct corresponds to the  \f$ log(r) \f$ kernel
    ///
    ///   The kernel \f$k(x,y): R^{1} -> R^{kn}\f$  with\f$ kn = km = 1\f$
    ///           \f$k(x,y) = \ln| x - y |\f$
    /// The kernel is non homogeneous
    ///
    struct ln_2d
    {
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};
        static constexpr auto symmetry_tag{symmetry::symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{1};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("ln_2d"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>{ValueType(1.)};
        }

        template<typename ValueType>
        [[nodiscard]] inline auto evaluate(container::point<ValueType, 2> const& x,
                                           container::point<ValueType, 2> const& y) const noexcept
        {
            using decayed_type = typename std::decay_t<ValueType>;

            return matrix_type<decayed_type>{xsimd::log(
              xsimd::sqrt((x.at(0) - y.at(0)) * (x.at(0) - y.at(0)) + (x.at(1) - y.at(1)) * (x.at(1) - y.at(1))))};
        }

        static constexpr int separation_criterion{1};
    };

    ///////
    /// \brief The grad ln_2d struct corresponds to the  \f$ ld/dr log(r) \f$ kernel
    ///
    ///   The kernel \f$k(x,y): R^{1} -> R^{2}\f$  with\f$ kn =2 ;  km = 1\f$
    ///           \f$k(x,y) = ( (x-y)/| x - y |)\f$
    /// The kernel is homogeneous with coefficient (1.0, 1.0)
    ///
    struct grad_ln_2d
    {
        static constexpr auto homogeneity_tag{homogeneity::homogenous};
        static constexpr auto symmetry_tag{symmetry::non_symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{2};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("grad_ln_2d"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>({ValueType(-1.), ValueType(-1.)});
        }
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, 2> const& x,
                                           container::point<ValueType2, 2> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = typename std::decay_t<ValueType1>;

            auto diff = x - y;
            decayed_type tmp = decayed_type(1.0) / (diff.at(0) * diff.at(0) + diff.at(1) * diff.at(1));
            return matrix_type<decayed_type>{tmp * diff.at(0), tmp * diff.at(1)};
        }

        template<typename ValueType>
        [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        {
            return vector_type<ValueType>{ValueType(1. / cell_width), ValueType(1. / cell_width)};
        }

        static constexpr int separation_criterion{1};
    };
    ///////
    /// \brief The val_grad ln_2d struct corresponds to the  \f$ ld/dr log(r) \f$ kernel
    ///
    ///   The kernel \f$k(x,y): R^{1} -> R^{kn}\f$  with\f$ kn =3 ;  km = 1\f$
    ///           \f$k(x,y) = (ln(| x - y |^), (x-y)/| x - y |^2)\f$
    /// The kernel is non homogeneous
    ///
    struct val_grad_ln_2d
    {
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};
        static constexpr auto symmetry_tag{symmetry::non_symmetric};
        static constexpr std::size_t km{1};
        static constexpr std::size_t kn{3};
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;

        const std::string name() const { return std::string("val_grad_ln_2d"); }

        template<typename ValueType>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType>({ValueType(1.), ValueType(-1.), ValueType(-1.)});
        }
        template<typename ValueType1, typename ValueType2>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, 2> const& x,
                                           container::point<ValueType2, 2> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            using decayed_type = typename std::decay_t<ValueType1>;
            auto diff = x - y;
            decayed_type norm = xsimd::sqrt(diff.at(0) * diff.at(0) + diff.at(1) * diff.at(1));
            decayed_type tmp = decayed_type(1.0) / norm;
            return matrix_type<decayed_type>{xsimd::log(norm), tmp * diff.at(0), tmp * diff.at(1)};
        }

        static constexpr int separation_criterion{1};
    };

}   // namespace scalfmm::matrix_kernels::laplace

#endif   // SCALFMM_MATRIX_KERNELS_LAPLACE_HPP
