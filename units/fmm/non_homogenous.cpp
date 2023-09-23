//
// Units for test fmm
// ----------------------
// @FUSE_FFTW
// @FUSE_CBLAS

#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/debug.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "units_fmm.hpp"
#include <string>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

using namespace scalfmm;
TEMPLATE_TEST_CASE("test fmm non homogenous", "[test-fmm]", double, float)
{
    using value_type = TestType;
    std::string path{TEST_DATA_FILES_PATH};
    using options_algo_type = scalfmm::options::seq_<>;

    SECTION("fmm test non homogenous", "[fmm-test-non_homogenous]")
    {
        constexpr int dim = 3;

        using matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_non_homogenous;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type =
          scalfmm::interpolation::interpolator<value_type, dim, matrix_kernel_type,
                                               scalfmm::options::uniform_<scalfmm::options::fft_>>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path.append("unitCube_3d_100_ref.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path, 4, 2, 5, true, options_algo_type{}));
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);

    return result;
}
