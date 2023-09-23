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
#include "units_source-target.hpp"

using namespace scalfmm;

TEMPLATE_TEST_CASE("test-source-target", "[test-source-target]", scalfmm::matrix_kernels::others::one_over_r2)
{
    using matrix_kernel_type = TestType;
    using options_type = scalfmm::options::uniform_<scalfmm::options::fft_>;
    std::string path_source{TEST_DATA_FILES_PATH};
    std::string path_target{TEST_DATA_FILES_PATH};

    // auto run(const std::string& input_source_file, const std::string& input_target_file, const int& tree_height,
    //          const int& group_size, const int& order, OptionsType op)
    // ->int
    SECTION("fmm test 2D double", "[fmm-test-2D-double]")
    {
        constexpr int dim = 2;
        using value_type = double;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path_source.append("circle-100_source.fma");
        path_target.append("circle-100_target.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path_source, path_target, 4, 3, 5, scalfmm::options::seq));
    }
    SECTION("fmm test 2D float", "[fmm-test-3D-float]")
    {
        constexpr int dim = 2;
        using value_type = float;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path_source.append("circle-100_source.fma");
        path_target.append("circle-100_target.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path_source, path_target, 4, 2, 5, scalfmm::options::seq));
    }
}
TEMPLATE_TEST_CASE("test-source-target-3D", "[test-source-target3D]", scalfmm::matrix_kernels::laplace::one_over_r)
{
    using matrix_kernel_type = TestType;
    using options_type = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;
    std::string path_source{TEST_DATA_FILES_PATH};
    std::string path_target{TEST_DATA_FILES_PATH};

    // auto run(const std::string& input_source_file, const std::string& input_target_file, const int& tree_height,
    //          const int& group_size, const int& order, OptionsType op)
    // ->int
    SECTION("fmm test 3D double", "[fmm-test-3D-double]")
    {
        constexpr int dim = 3;
        using value_type = double;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path_source.append("sphere-706_source.fma");
        path_target.append("sphere-706_target.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path_source, path_target, 4, 2, 5, scalfmm::options::seq));
    }
    SECTION("fmm test 3D float", "[fmm-test-3D-float]")
    {
        constexpr int dim = 3;
        using value_type = float;

        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        path_source.append("sphere-706_source.fma");
        path_target.append("sphere-706_target.fma");
        REQUIRE(run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(
          path_source, path_target, 5, 7, 5, scalfmm::options::seq));
    }
}
int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);

    return result;
}
