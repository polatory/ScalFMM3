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

TEST_CASE("test-grad-one-over-r", "[test-fmm-grad-one-over-r]")
{
    using value_type = double;
    using options_algo_type = scalfmm::options::seq_<>;
    using options_type = scalfmm::options::uniform_<scalfmm::options::low_rank_>;
    std::string path{TEST_DATA_FILES_PATH};

    SECTION("fmm test 1D", "[fmm-test-1D]")
    {
        constexpr int dim = 1;

        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<1>;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_1d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, options_algo_type{}));
    }
    SECTION("fmm test 2D", "[fmm-test-2D]")
    {
        constexpr int dim = 2;

        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<2>;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_2d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, options_algo_type{}));
    }
    SECTION("fmm test 3D", "[fmm-test-3D]")
    {
        constexpr int dim = 3;

        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<3>;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_3d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 6, true, options_algo_type{}, 5));
    }
    SECTION("fmm test 4D", "[fmm-test-4D]")
    {
        constexpr int dim = 4;

        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<4>;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_4d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 3, true, options_algo_type{}));
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);

    return result;
}
