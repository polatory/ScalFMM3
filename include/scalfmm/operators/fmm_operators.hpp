// --------------------------------
// See LICENCE file at project root
// File : operators/fmm_operators.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_FMM_OPERATORS_HPP
#define SCALFMM_OPERATORS_FMM_OPERATORS_HPP

#include <cstdlib>
#include <iostream>

namespace scalfmm::operators
{
    ///
    /// /
    /// \class near_field_operator
    /// \brief The near_field_operator class
    ///
    /// This class concerns the near field operator for the fmm operators.
    ///    It is used in direct pass in all algorithms.
    ///    In general, it is enough to only have the matrix kernel member.
    ///  @tparam MATRIX_KERNEL_TYPE template for matrix kernel class
    ///
    ///   The following example constructs the near field operator to compute the potential and the
    ///     force associated to the potential 1/r
    /// \code
    ///   using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r;
    ///
    /// \endcode
    template<typename MatrixKernel>
    class near_field_operator
    {
      public:
        using matrix_kernel_type = MatrixKernel;
        ///
        /// \brief near_field_operator constructor
        /// \param[in] matrix_kernel -  the matrix kernel used int the near field
        ///
        // near_field_operator() = delete;
        near_field_operator(near_field_operator const&) = delete;
        near_field_operator(near_field_operator&&) = delete;
        auto operator=(near_field_operator const&) -> near_field_operator& = default;
        auto operator=(near_field_operator&&) -> near_field_operator& = delete;
        ///
        /// \brief near_field_operator constructor
        ///  The separation_criterion is the matrix_kernel one and mutual is set to true
        ///   and mutual is set to true
        ///
        near_field_operator()
          : m_matrix_kernel(matrix_kernel_type())
          , m_mutual(true)
        {
            m_separation_criterion = m_matrix_kernel.separation_criterion;
        }
        // ///
        /// \brief near_field_operator constructor
        ///  The separation_criterion is the matrix_kernel one and mutual is set to true and
        //
        ///
        /// \param[in] mutual  to specify if in p2p we use mutual interaction
        near_field_operator(bool mutual)
          : m_matrix_kernel(matrix_kernel_type())
          , m_mutual(mutual)
        {
            m_separation_criterion = m_matrix_kernel.separation_criterion;
        }
        ///
        /// \brief near_field_operator constructor
        ///  The separation_criterion is the matrix_kernel one and mutual is set to true
        /// and
        ///
        /// \param[in] matrix_kernel the matrix kernel used int the near field
        near_field_operator(matrix_kernel_type const& matrix_kernel)
          : m_matrix_kernel(matrix_kernel)
          , m_separation_criterion(matrix_kernel.separation_criterion)
          , m_mutual(true)
        {
        }

        ///
        /// \brief near_field_operator constructor
        ///
        ///  The separation_criterion is the matrix_kernel one
        ///
        /// \param[in] matrix_kernel the matrix kernel used int the near field
        /// \param[in] mutual  the boolean to specify if the P2P is symmetric or not (Mutual interaction)
        ///
        near_field_operator(matrix_kernel_type const& matrix_kernel, const bool mutual)
          : m_matrix_kernel(matrix_kernel)
          , m_separation_criterion(matrix_kernel.separation_criterion)
          , m_mutual(mutual)
        {
        }
        ///
        /// \brief matrix_kernel accessor
        /// @return the matrix kernel operator
        ///
        auto matrix_kernel() const -> matrix_kernel_type const& { return m_matrix_kernel; }

        /// @brief matrix_kernel accessor
        /// @return
        auto matrix_kernel() -> matrix_kernel_type& { return m_matrix_kernel; }

        ///
        /// \brief separation_criterion accessor
        /// @return the separation criteria used in matrix kernel
        ///
        auto separation_criterion() const -> int { return m_separation_criterion; }
        auto separation_criterion() -> int& { return m_separation_criterion; }

        ///
        /// \brief The accessor to specify if we are using the symmetric P2P (mutual interactions)
        /// @return the separation criteria used in matrix kernel
        ///
        auto mutual() const -> bool { return m_mutual; }
        auto mutual() -> bool& { return m_mutual; }

      private:
        matrix_kernel_type m_matrix_kernel;   ///<  the matrix kernel used in the near field
        int m_separation_criterion{
          matrix_kernel_type::separation_criterion};   ///<  the separation criterion; used int the near field
        bool m_mutual{true};   ///<  To specify if you use symmetric algorithm for the p2p (Mutual interactions)
    };

    ///
    ///
    /// \class far_field_operator
    /// \brief The far_field_operator class
    ///
    /// This class concerns the far field operator for the fmm operators.
    ///
    ///  @tparam APPROXIMATION_TYPE template for approximation method used to compute the far field
    ///
    ///   The following example constructs the far field operator based on the uniform interpolation to compute
    ///      the potential associated to the potential 1/r
    ///
    /// \code
    ///     using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    ///     using interpolation_type = scalfmm::interpolation::uniform_interpolator<double, dimension,
    ///     matrix_kernel_type>;
    ///     using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
    ///
    /// \endcode
    template<typename Approximation, bool ComputeGradient = false>
    class far_field_operator
    {
      public:
        using approximation_type = Approximation;
        static constexpr bool compute_gradient = ComputeGradient;
        // precise the homogeneous type of the kernel
        static constexpr auto homogeneity_tag = approximation_type::homogeneity_tag;

        far_field_operator() = delete;
        far_field_operator(far_field_operator const&) = delete;
        far_field_operator(far_field_operator&&) = delete;
        auto operator=(far_field_operator const&) -> far_field_operator& = default;
        auto operator=(far_field_operator&&) -> far_field_operator& = delete;
        ///
        /// \brief near_field_operator constructor
        /// \param[in] matrix_kernel -  the matrix kernel used int the near field
        ///
        far_field_operator(approximation_type const& approximation_method)
          : m_approximation(approximation_method)
        {
        }

        ///
        /// \brief approximation method  accessor
        /// @return the approximation method
        ///
        auto approximation() const -> approximation_type const& { return m_approximation; }
        ///
        ///
        /// \brief separation_criterion accessor
        /// @return the separation criteria used in matrix kernel
        ///
        auto separation_criterion() const -> int { return m_separation_criterion; }
        auto separation_criterion() -> int& { return m_separation_criterion; }
        ///
      private:
        // Here, be careful with this reference with the lifetime of matrix_kernel.
        approximation_type const& m_approximation;   ///<  the approximation method used in the near field
        int m_separation_criterion{approximation_type::separation_criterion};
    };
    ///
    /// \class fmm_operators
    /// \brief The fmm_operators class
    ///
    ///
    ///  @tparam NEAR_FIELD_TYPE template for the near field operator
    ///  @tparam FAR_FIELD_TYPE template for the far field operator
    ///
    ///  The following example constructs the near field operator to compute the potential associated to the potential
    ///  1/r
    ///
    /// \code
    ///   // Near field
    ///   using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r;
    ///   using nar_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
    ///   // Far field is the uniform interpolation
    ///   using far_field_type = scalfmm::interpolation::uniform_interpolator<double, dimension, matrix_kernel_type>;
    ///    using fmm_operator = scalfmm::operators::fmm_operators<near_field_type, far_field_type>  ;
    /// \endcode
    ///
    /// The following example constructs the near-field operator to calculate the potential and
    ///  the force associated with the potential 1/r.  The far field considers the
    ///  one_over_r matrix kernel and calculates the force by deriving the interpolation polynomial
    /// of the potential.
    ///
    /// \code
    ///   using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r;
    ///   using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
    ///   //
    ///   using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    ///   // Far field is the uniform interpolation
    ///   using far_field_type = scalfmm::interpolation::uniform_interpolator<double, dimension,
    ///   far_matrix_kernel_type>; using fmm_operator = scalfmm::operators::fmm_operators<near_field_type,
    ///   far_field_type>  ;
    /// \endcode
    template<typename NearField, typename FarField>
    class fmm_operators
    {
      public:
        using near_field_type = NearField;
        using far_field_type = FarField;

        fmm_operators() = delete;
        fmm_operators(fmm_operators&&) = delete;
        fmm_operators(fmm_operators const& other) = delete;
        auto operator=(fmm_operators const&) -> fmm_operators& = delete;
        auto operator=(fmm_operators&&) -> fmm_operators& = delete;

        fmm_operators(near_field_type const& near_field, far_field_type const& far_field)
          : m_near_field(near_field)
          , m_far_field(far_field)
        {
            if(m_near_field.separation_criterion() != m_far_field.separation_criterion())
            {
                std::cerr << "The separation criteria is not the same in the near "
                             "and far fields !!"
                          << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        /// \brief near field accessor
        /// @return the near field operator
        ///
        auto near_field() const -> near_field_type const& { return m_near_field; }
        /// \brief far field accessor
        /// @return the ar field operator
        ///
        auto far_field() const -> far_field_type const& { return m_far_field; }

      private:
        near_field_type const& m_near_field;   ///< the near field used int the fmm operator
        far_field_type const& m_far_field;     ///< the far field used int the fmm operator
    };

}   // namespace scalfmm::operators

#endif
