// @FUSE_MPI
//
// Units for test fmm
// ----------------------

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

static cpp_tools::parallel_manager::parallel_manager para;

#include "units_count_kernel_mpi_gen.hpp"

//auto run(const int& tree_height, const int& group_size,  std::string& input_file,  bool use_leaf_distribution, bool mutual = false) -> int{

TEMPLATE_TEST_CASE("test count 2d", "[test-count-2d]", double)
{
    // leaf distrubution and not mutual
    SECTION("count 2d", "[count2d]")
    {
        run_count_kernel_mpi<2, double>(4, 10, path + "test_2d_ref.fma", true, false);
    }   // h = 5
}

TEMPLATE_TEST_CASE("test count 3d", "[test-count-3d]", double)
{
    // leaf distribution and mutual
    SECTION("count 3d", "[count3d]")
    {
        run_count_kernel_mpi<3, double>(5, 40, path + "sphere-706_source.fma", true, true);
    }   // h = 5
}

TEMPLATE_TEST_CASE("test count 3d", "[test-count-3d]", float)
{
    // leaf distribution and mutual
    SECTION("count 3d", "[count3d]")
    {
        run_count_kernel_mpi<3, float>(5, 40, path + "sphere-706_source.fma", true, true);
    }   // h = 5
}
int main(int argc, char* argv[])
{
    para.init();
    int result = Catch::Session().run(argc, argv);
    para.end();

    return result;
}
