//
// Units for test fmm
// ----------------------
// @FUSE_FFTW
// @FUSE_CBLAS

#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "units_fmm.hpp"
#include <string>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

using namespace scalfmm;

TEST_CASE("test-shift-val-grad-one-over-r", "[test-fmm-shift-val-grad-one-over-r]")
{
    using value_type = double;
    using options_algo_type = scalfmm::options::omp_<>;
    using options_type = scalfmm::options::uniform_<scalfmm::options::fft_>;
    std::string path{TEST_DATA_FILES_PATH};

    SECTION("fmm test 1D", "[fmm-test-1D]")
    {
        constexpr int dim = 1;

        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dim>;
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, far_matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;
        path.append("test_1d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, options_algo_type{}));
    }
    SECTION("fmm test 2D", "[fmm-test-2D]")
    {
        constexpr int dim = 2;

        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dim>;
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, far_matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;
        path.append("test_2d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, options_algo_type{}));
    }
    SECTION("fmm test 3D", "[fmm-test-3D]")
    {
        constexpr int dim = 3;

        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dim>;
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, far_matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;
        path.append("test_3d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, options_algo_type{}, 7));
    }
}

TEST_CASE("test-shift-grad-one-over-r", "[test-fmm-shift-grad-one-over-r]")
{
    using value_type = double;
    using options_algo_type = scalfmm::options::seq_<>;
    using options_type = scalfmm::options::uniform_<scalfmm::options::dense_>;
    std::string path{TEST_DATA_FILES_PATH};

    SECTION("fmm test 1D", "[fmm-test-1D]")
    {
        constexpr int dim = 1;

        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dim>;
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, far_matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;
        path.append("test_1d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 10, 5, true, options_algo_type{}));
    }
    SECTION("fmm test 2D", "[fmm-test-2D]")
    {
        constexpr int dim = 2;

        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dim>;
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, far_matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;
        path.append("test_2d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 10, 5, true, options_algo_type{}));
    }
    SECTION("fmm test 3D", "[fmm-test-3D]")
    {
        constexpr int dim = 3;

        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dim>;
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, far_matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;
        path.append("test_3d_ref.fma");
        std::cout << " fmm test 3D \n";
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, options_algo_type{}, 7));
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);

    return result;
}
