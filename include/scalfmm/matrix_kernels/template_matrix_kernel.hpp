#ifndef SCALFMM_MATRIX_KERNELS_TEMPLATE_HPP
#define SCALFMM_MATRIX_KERNELS_TEMPLATE_HPP
///////
/// \brief The name struct corresponds to the the name of the kernel \f$ K(x,y) \f$ 
///
///   The kernel \f$K(x,y): R^{km} -> R^{kn}\f$
///
/// The kernel is homogeneous if K(ax,ay) = a^p K(x,y) holds
/// The kernel is symmetric if the kernel satisfies all symmetries (axes, x=y, ...)
///
struct name
{
    static constexpr auto homogeneity_tag{};   // homogeneity::homogenous or homogeneity::non_homogenous
    static constexpr auto symmetry_tag{};      // symmetry::symmetric or symmetry::non_symmetric
    static constexpr std::size_t km{};         // the dimension
    static constexpr std::size_t kn{};
    /**
     * @brief
     *
     */
    static constexpr int separation_criterion{1}; // the separation criterion used to separate near and far field.
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
    const std::string name() const { return std::string("name"); }

    template<typename ValueType>
    /**
     * @brief Return the mutual coefficient of size kn
     *
     * The coefficient is used in the direct pass when the kernel is used
     *  to compute the interactions inside the leaf when we use the symmetry
     *  of tke kernel ( N^2/2 when N is the number of particles)
     *
     * @return constexpr auto
     */
    [[nodiscard]] inline constexpr auto mutual_coefficient() const
    {
        return vector_type<ValueType>{ValueType(1.)};
    }

    template<typename ValueType>
    /**
     * @brief evaluate the kernel at points x and y
     *
     *
     * @param x 2d point
     * @param y 2d point
     * @return  return the matrix K(x,y) in a vector (row storage)
     */
    template<typename ValueType1, typename ValueType2, int Dim>
    [[nodiscard]] inline auto evaluate(container::point<ValueType1, 2> const& x,
                                       container::point<ValueType2, 2> const& y) const noexcept
      -> std::enable_if_t<std::is_same_v<std::decay_t<ValueType1>, std::decay_t<ValueType2>>,
                          matrix_type<std::decay_t<ValueType1>>>
    {
        using decayed_type = typename std::decay_t<ValueType1>;

        return matrix_type<decayed_type>{...};
    }
    /**
     * @brief return the scale factor of the kernel
     *
     * the method is used only if the kernel is homogeneous
     */
    template<typename ValueType>
    [[nodiscard]] inline auto scale_factor(ValueType cell_width) const noexcept
    {
        return vector_type<ValueType>{...};
    }
};
#endif
