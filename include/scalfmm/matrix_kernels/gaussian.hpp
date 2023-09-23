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
        static constexpr auto symmetry_tag{symmetry::symmetric};   // symmetry::symmetric or symmetry::non_symmetric
        static constexpr std::size_t km{1};                        // the dimension
        static constexpr std::size_t kn{1};
        /**
         * @brief
         *
         */
        const std::size_t separation_criterion{0};   // the separation criterion used to separate near and far field.
        ValueType m_coeff{ValueType(1.)};

        /**
         * @brief Set the coeff object
         *
         * @param inCoeff
         */
        set_coeff(ValueType inCoeff) { coeff = inCoeff; }
        //
        // Mandatory type
        template<typename ValueType>
        using matrix_type = std::array<ValueType, kn * km>;
        template<typename ValueType>
        using vector_type = std::array<ValueType, kn>;
        //
        /**
         * @brief return the name of the kernel
         *
         */
        const std::string name() const { return std::string("gaussian "); }

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
        // [[nodiscard]] inline constexpr auto mutual_coefficient() const
        // {
        //     return vector_type<ValueType>{ValueType(1.)};
        // }

        /**
         * @brief evaluate the kernel at points x and y
         *
         *
         * @param x d point
         * @param y d point
         * @return  return the matrix K(x,y)
         */
        template<typename ValueType1, typename ValueType2, int Dim>
        [[nodiscard]] inline auto evaluate(container::point<ValueType1, 2> const& x,
                                           container::point<ValueType2, 2> const& y) const noexcept
          -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                              matrix_type<std::decay_t<ValueType1>>>
        {
            return variadic_evaluate(x, y, std::make_index_sequence<Dim>{});
        }
        template<ypename ValueType1, typename ValueType2, std::size_t... Is>
        [[nodiscard]] inline auto variadic_evaluate(container::point<ValueType1, Dim> const& xs,
                                                    container::point<ValueType2, Dim> const& ys,
                                                    std::index_sequence<Is...> is) const noexcept
        {
            using decayed_type = std::decay_t<ValueType1>,
                  decayed_type r2 = (((xs.at(Is) - ys.at(Is)) * (xs.at(Is) - ys.at(Is))) + ...);
            decayed_type r3{xsimd::pow(tmp, int(3))};
            return matrix_type<Valdecayed_typeueType>{r2 * coeff};
            //-r3 * (xs - ys);
        }
        // /**
        //  * @brief return the scale factor of the kernel
        //  *
        //  * the method is used only if the kernel is homogeneous
        //  */
        // template<typename ValueType>
        // [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
        // {
        //     return vector_type<ValueType>{...};
        // }
    };
}   // namespace scalfmm::matrix_kernels
#endif
