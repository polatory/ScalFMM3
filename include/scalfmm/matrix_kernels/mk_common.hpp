#ifndef SCALFMM_MATRIX_KERNELS_MK_COMMON_HPP
#define SCALFMM_MATRIX_KERNELS_MK_COMMON_HPP

namespace scalfmm::matrix_kernels
{
    /**
     * @brief specify if the kernel is homogeneous or not
     *
     */
    enum struct homogeneity
    {
        homogenous,  
        non_homogenous
    };
    /**
     * @brief specify if the kernel is symmetric or not
     *
     */
    enum struct symmetry
    {
        symmetric,
        //   antisymmetric,
        non_symmetric
    };

    /**
     * @brief describe the convergence of the expansion at level 0
     *    in the periodic case
     *
     *
     */
    enum struct periodicity
    {
        absolutely,   //< absolutely convergent series
        condionally   //< conditionally convergent series
    };

}   // namespace scalfmm::matrix_kernels
#endif
