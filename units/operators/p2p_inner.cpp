//
// Units for l2p operator
// ----------------------
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/uniform.hpp"
#include "scalfmm/kernels/generic/p2p.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/is_valid.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/leaf.hpp"
#include "scalfmm/utils/generate.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

using namespace scalfmm;

template<typename ValueType, std::size_t dimension, std::size_t physical_values>
void test_p2p_inner(std::size_t order)
{
    using value_type = ValueType;
    using pack_of_physical_values = meta::pack<physical_values, value_type>;
    using particle_type = typename utils::get_particle_type<value_type, dimension, physical_values>::type;
    // td<particle_type> t;
    using outputs_type = meta::pack_expand_tuple<pack_of_physical_values>;   // std::tuple<value_type>;
    using output_vector = container::variadic_container_tuple<outputs_type>;
    // td<outputs_type> t1;
    using leaf_type = component::leaf<particle_type, outputs_type>;
    using interpolator_type =
      interpolation::uniform_interpolator<matrix_kernels::laplace::one_over_r, value_type, dimension>;

    const std::size_t tree_height{3};
    const value_type box_width{1.};
    interpolator_type uniform(order, tree_height, box_width);

    container::point<value_type, dimension> center{utils::get_center<value_type, dimension>()};

    const value_type width{0.25};

    const std::size_t nb_particles{13};
    auto container{utils::generate_particles<particle_type>(nb_particles, center, width)};
    output_vector outputs(nb_particles);

    leaf_type leaf(std::cbegin(container), std::cend(container), std::get<0>(container.size()), center, width, 0);

    operators::p2p_inner(uniform, leaf);
    utils::full_direct_test<value_type, dimension, physical_values>(std::begin(container), std::end(container),
                                                                    uniform.matrix_kernel(), outputs);

    auto const& output = leaf.coutputs();
    auto container_it = std::cbegin(outputs);

    std::for_each(std::begin(output), std::end(output), [&container_it](auto const& p) {
        REQUIRE(utils::meta_compare(p, *container_it, [](auto const& a, auto const& b) {
            return std::fabs(a - b) <= std::numeric_limits<value_type>::epsilon() * 5;
            // return utils::almost_equal(a, b, std::numeric_limits<value_type>::digits10);
        }));
        ++container_it;
    });
}

TEMPLATE_TEST_CASE("p2p-inner-test-3D-1PV", "[p2p-inner-test-3D-1PV]", float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=3, pv=1", "[p2p-inner,d=3,pv=1]")
    {
        test_p2p_inner<value_type, 3, 1>(2);
        test_p2p_inner<value_type, 3, 1>(7);
    }
}

TEMPLATE_TEST_CASE("p2p-inner-test-2D-1PV", "[p2p-inner-test-2D-1PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=2, pv=1", "[p2p-inner,d=2,pv=1]")
    {
        test_p2p_inner<value_type, 2, 1>(2);
        test_p2p_inner<value_type, 2, 1>(7);
    }
}

// TEMPLATE_TEST_CASE("p2p-inner-test-1D-1PV", "[p2p-inner-test-1D-1PV]", double, float)
//{
//    using value_type = TestType;
//
//    SECTION("p2p inner with d=1, pv=1", "[p2p-inner,d=1,pv=1]")
//    {
//        test_p2p_inner<value_type, 1, 1>(2);
//        test_p2p_inner<value_type, 1, 1>(7);
//    }
//}

TEMPLATE_TEST_CASE("p2p-inner-test-3D-2PV", "[p2p-inner-test-3D-2PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=3, pv=2", "[p2p-inner,d=3,pv=2]")
    {
        test_p2p_inner<value_type, 3, 2>(2);
        test_p2p_inner<value_type, 3, 2>(7);
    }
}

TEMPLATE_TEST_CASE("p2p-inner-test-2D-3PV", "[p2p-inner-test-2D-3PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=2, pv=3", "[p2p-inner,d=2,pv=3]")
    {
        test_p2p_inner<value_type, 2, 3>(3);
        test_p2p_inner<value_type, 2, 3>(6);
    }
}

// TEMPLATE_TEST_CASE("p2p-inner-test-1D-4PV", "[p2p-inner-test-1D-4PV]", double, float)
//{
//    using value_type = TestType;
//
//    SECTION("p2p inner with d=1, pv=4", "[p2p-inner,d=1,pv=4]")
//    {
//        test_p2p_inner<value_type, 1, 4>(2);
//        test_p2p_inner<value_type, 1, 4>(7);
//    }
//}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
