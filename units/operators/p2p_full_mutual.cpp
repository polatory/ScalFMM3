//
// Units for l2p operator
// ----------------------
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/container/variadic_adaptor.hpp"
#include "scalfmm/interpolation/uniform.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/leaf.hpp"
#include "scalfmm/utils/generate.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <scalfmm/tools/colorized.hpp>

#include "scalfmm/kernels/generic/p2p.hpp"

using namespace scalfmm;

template<typename ValueType, std::size_t dimension, std::size_t physical_values>
auto test_p2p_full_mutual(std::size_t order) -> void
{
    using value_type = ValueType;
    using pack_of_physical_values = meta::pack<physical_values, value_type>;
    using particle_type = typename utils::get_particle_type<value_type, dimension, physical_values>::type;
    using outputs_type = meta::pack_expand_tuple<pack_of_physical_values>;   // std::tuple<value_type>;
    using output_vector = container::variadic_container_tuple<outputs_type>;
    using leaf_type = component::leaf<particle_type, outputs_type>;
    using interpolator_type =
      interpolation::uniform_interpolator<scalfmm::matrix_kernels::laplace::one_over_r, value_type, dimension>;

    const std::size_t tree_height{3};
    const value_type box_width{1.};
    interpolator_type uniform(order, tree_height, box_width);

    const value_type width{0.25};
    container::point<value_type, dimension> center_f1{utils::get_center<value_type, dimension>()};
    auto center_f2 = center_f1 + container::point<value_type, dimension>{-width, -width, 0.};

    const std::size_t nb_particles{5};
    auto container_f1{utils::generate_particles<particle_type>(nb_particles, center_f1, width)};
    auto container_f2{utils::generate_particles<particle_type>(nb_particles, center_f2, width)};
    decltype(container_f1) container_f0(nb_particles * 2);
    auto begin_next = std::copy(std::cbegin(container_f1), std::cend(container_f1), std::begin(container_f0));
    begin_next = std::copy(std::cbegin(container_f2), std::cend(container_f2), begin_next);

    output_vector outputs_direct_f1(nb_particles);
    output_vector outputs_direct_f2(nb_particles);
    output_vector outputs_direct_f0(nb_particles * 2);

    utils::full_direct_test<value_type, dimension, physical_values>(std::begin(container_f0), std::end(container_f0),
                                                                    uniform.matrix_kernel(), outputs_direct_f0);

    std::vector<std::shared_ptr<leaf_type>> leaf_f1 = {std::make_shared<leaf_type>(
      std::cbegin(container_f1), std::cend(container_f1), std::get<0>(container_f1.size()), center_f1, width, 0)};

    std::vector<std::shared_ptr<leaf_type>> leaf_f2 = {std::make_shared<leaf_type>(
      std::cbegin(container_f2), std::cend(container_f2), std::get<0>(container_f2.size()), center_f2, width, 1)};

    operators::p2p_full_mutual(uniform, *(leaf_f1[0]), leaf_f2);
    operators::p2p_inner(uniform, *(leaf_f1[0]));
    operators::p2p_inner(uniform, *(leaf_f2[0]));

    auto const& outputs_f1 = leaf_f1[0]->coutputs();
    auto const& outputs_f2 = leaf_f2[0]->coutputs();
    auto container_it = std::cbegin(outputs_direct_f0);

    std::for_each(std::begin(outputs_f1), std::end(outputs_f1), [&container_it](auto const& p) {
        REQUIRE(utils::meta_compare(p, *container_it,
                                    [](auto const& a, auto const& b) { return utils::almost_equal(a, b, 7); }));
        ++container_it;
    });
    std::for_each(std::begin(outputs_f2), std::end(outputs_f2), [&container_it](auto const& p) {
        REQUIRE(utils::meta_compare(p, *container_it,
                                    [](auto const& a, auto const& b) { return utils::almost_equal(a, b, 7); }));
        ++container_it;
    });
}

TEMPLATE_TEST_CASE("p2p-inner-test-3D-1PV", "[p2p-inner-test-3D-1PV]", double)   //, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=3, pv=1", "[p2p-inner,d=3,pv=1]")
    {
        test_p2p_full_mutual<value_type, 3, 1>(2);
        test_p2p_full_mutual<value_type, 3, 1>(7);
    }
}

TEMPLATE_TEST_CASE("p2p-inner-test-2D-1PV", "[p2p-inner-test-2D-1PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=2, pv=1", "[p2p-inner,d=2,pv=1]")
    {
        test_p2p_full_mutual<value_type, 2, 1>(2);
        test_p2p_full_mutual<value_type, 2, 1>(7);
    }
}

TEMPLATE_TEST_CASE("p2p-inner-test-1D-1PV", "[p2p-inner-test-1D-1PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=1, pv=1", "[p2p-inner,d=1,pv=1]")
    {
        test_p2p_full_mutual<value_type, 1, 1>(2);
        test_p2p_full_mutual<value_type, 1, 1>(7);
    }
}

TEMPLATE_TEST_CASE("p2p-inner-test-3D-2PV", "[p2p-inner-test-3D-2PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=3, pv=2", "[p2p-inner,d=3,pv=2]")
    {
        test_p2p_full_mutual<value_type, 3, 2>(2);
        test_p2p_full_mutual<value_type, 3, 2>(7);
    }
}

TEMPLATE_TEST_CASE("p2p-inner-test-2D-3PV", "[p2p-inner-test-2D-3PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=2, pv=3", "[p2p-inner,d=2,pv=3]")
    {
        test_p2p_full_mutual<value_type, 2, 3>(3);
        test_p2p_full_mutual<value_type, 2, 3>(6);
    }
}

TEMPLATE_TEST_CASE("p2p-inner-test-1D-4PV", "[p2p-inner-test-1D-4PV]", double, float)
{
    using value_type = TestType;

    SECTION("p2p inner with d=1, pv=4", "[p2p-inner,d=1,pv=4]")
    {
        test_p2p_full_mutual<value_type, 1, 4>(2);
        test_p2p_full_mutual<value_type, 1, 4>(7);
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
