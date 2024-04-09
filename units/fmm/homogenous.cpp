// Units for test fmm
// ----------------------
// @FUSE_FFTW
// @FUSE_CBLAS

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "units_fmm.hpp"

using namespace scalfmm;

TEMPLATE_TEST_CASE("test-homogenous", "[test-homogenous]", scalfmm::matrix_kernels::others::one_over_r2,
                   scalfmm::matrix_kernels::laplace::one_over_r)
{
    using value_type = double;
    using matrix_kernel_type = TestType;
    using options_type = scalfmm::options::chebyshev_<scalfmm::options::dense_>;
    std::string path{TEST_DATA_FILES_PATH};

    SECTION("fmm test 1D", "[fmm-test-1D]")
    {
        constexpr int dim = 1;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_1d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, scalfmm::options::seq));
    }
    SECTION("fmm test 2D", "[fmm-test-2D]")
    {
        constexpr int dim = 2;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_2d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, scalfmm::options::seq));
    }
    SECTION("fmm test 3D", "[fmm-test-3D]")
    {
        constexpr int dim = 3;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_3d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, scalfmm::options::seq));
    }
    SECTION("fmm test 4D", "[fmm-test-4D]")
    {
        constexpr int dim = 4;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("test_4d_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 3, true, scalfmm::options::seq));
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);

    return result;
}
