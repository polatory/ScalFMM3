#ifndef SCALFMM_MATRIX_KERNELS_GAUSSIAN_HPP
#define SCALFMM_MATRIX_KERNELS_GAUSSIAN_HPP

#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/mk_common.hpp>

namespace scalfmm::matrix_kernels
{

    ///////
    /// \brief The name struct corresponds to the  \f$ K(x,y) = exp(-|x-y|/(2 sigma^2)) \f$ kernel
    ///
    ///   The kernel \f$K(x,y): R^{km} -> R^{kn}\f$
    ///
    /// The kernel is not homogeneous K(ax,ay) != a^p K(x,y)
    /// The kernel is symmetric
    ///
    template<typename ValueType>
    struct gaussian
    {
        static constexpr auto homogeneity_tag{homogeneity::non_homogenous};
        static constexpr auto symmetry_tag{symmetry::non_symmetric};   // symmetry::symmetric or symmetry::non_symmetric
        static constexpr std::size_t km{1};                        // the dimension
        static constexpr std::size_t kn{1};
        /**
         * @brief
         *
         */
        ValueType m_coeff{ValueType(1.)};

        /**
         * @brief Set the coeff object
         *
         * @param inCoeff
         */
        void set_coeff(ValueType inCoeff) { m_coeff = inCoeff; }
        //
        // Mandatory type
        template<typename ValueType1>
        using matrix_type = std::array<ValueType1, kn * km>;
        template<typename ValueType1>
        using vector_type = std::array<ValueType1, kn>;
        //
        /**
         * @brief return the name of the kernel
         *
         */
        const std::string name() const { return std::string("gaussian ") + "  coeff = " + std::to_string(m_coeff); }

        // template<typename ValueType>
        // /**
        //  * @brief Return the mutual coefficient of size kn
        //  *
        //  * The coefficient is used in the direct pass when the kernel is used
        //  *  to compute the interactions inside the leaf when we use the symmetry
        //  *  of tke kernel ( N^2/2 when N is the number of particles)
        //  *
        //  * @return constexpr auto
        //  */
        template<typename ValueType1>
        [[nodiscard]] inline constexpr auto mutual_coefficient() const
        {
            return vector_type<ValueType1>{ValueType1(1.)};
        }

        /**
         * @brief evaluate the kernel at points x and y
         *
         *
         * @param x d point
         * @param y d point
         * @return  return the matrix K(x,y)
         */
        template<typename PointType1, typename PointType2>
        [[nodiscard]] inline auto evaluate(PointType1 const& x, PointType2 const& y) const noexcept -> std::enable_if_t<
          std::is_same_v<std::decay_t<typename PointType1::value_type>, std::decay_t<typename PointType2::value_type>>,
          matrix_type<std::decay_t<typename PointType1::value_type>>>
        {
            return variadic_evaluate(x, y, std::make_index_sequence<PointType1::dimension>{});
        }
        template<typename PointType1, typename PointType2, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(PointType1 const& xs, PointType2 const& ys,
                                                    std::index_sequence<Is...> is) const noexcept
        {
            using decayed_type = std::decay_t<typename PointType1::value_type>;
            auto l2 = decayed_type(-1.0) / (m_coeff * m_coeff);
            decayed_type r2 = (((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...);
            return matrix_type<decayed_type>{xsimd::exp(r2 * l2)};
        }
        /**
         * @brief return the scale factor of the kernel
         *
         * the method is used only if the kernel is homogeneous
         */
        // template<typename ValueType>
        // [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        // {
        //     return vector_type<ValueType>{...};
        // }

        static constexpr int separation_criterion{1};
    };
}   // namespace scalfmm::matrix_kernels
#endif
