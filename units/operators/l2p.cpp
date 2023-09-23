//
// Units for l2p operator
// ----------------------
// @FUSE_FFTW
#include "scalfmm/operators/l2p.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/is_valid.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
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
#include <xtensor/xeval.hpp>
#include <xtensor/xmanipulation.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

using namespace scalfmm;

template<typename FmmOperator, typename ValueType>
void test_l2p(std::size_t order)
{
    using value_type = ValueType;
    //
    using fmm_operator_type = FmmOperator;
    using far_field_type = typename fmm_operator_type::far_field_type;
    using near_field_type = typename fmm_operator_type::near_field_type;
    using interpolator_type = typename far_field_type::approximation_type;
    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
    using near_matrix_kernel_type = typename near_field_type::matrix_kernel_type;
    static constexpr std::size_t nb_inputs{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs{near_matrix_kernel_type::kn};
    static constexpr std::size_t dimension{interpolator_type::dimension};

    //
    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs, std::size_t>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using leaf_type = scalfmm::component::leaf<particle_type>;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;

    const std::size_t tree_height{3};
    const value_type box_width{1.};

    // We construct an fmm_operator for the call to the l2p operator.
    // This will allow us to distinguish optimized version of the l2p operator for
    // specific laplace kernels combination.
    interpolator_type interpolator(far_matrix_kernel_type{}, order, tree_height, box_width);
    far_field_type far_field(interpolator);
    //
    container::point<value_type, dimension> center{utils::get_center<value_type, dimension>()};


    //Dimension of the test box
    const value_type width{0.25};
    const auto half_width{width * 0.5};

    // We take 13 particles to test simd and scalar version of the operator
    const std::size_t nb_particles{13};

    //This function generate a container holding random particles in the box(center,width)
    // outputs are set to zero in the function.
    auto container{utils::generate_particles<particle_type>(nb_particles, center, width)};
    // We create the leaf corresponding to the box.
    leaf_type leaf(std::cbegin(container), std::cend(container), container.size(), center, width, 0);

    // Then the cell
    cell_type cell(center, width, order);

    auto roots = interpolator.roots();
    // Here we generate a grid of points corresponding to the roots in the targeted dimension.
    auto grid = tensor::generate_meshgrid<dimension>(roots);

    // We evaluate the meshgrid to get the points (i.e we evaluate the xtensor expression returned by the funtion
    // generate_meshgrid<>())
    auto eval_grid =
      std::apply([](auto&&... xs) { return std::make_tuple(xt::eval(std::forward<decltype(xs)>(xs))...); }, grid);

    // Then we get a flatten view the grid in each dimension.
    auto flatten_views = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::flatten(std::forward<decltype(xs)>(xs))...); }, eval_grid);

    /// We move the point of the grid to match our test box. (i.e we apply the transformation on the flatten view)
    meta::for_each(flatten_views, flatten_views, center,
                   [&half_width](auto const& g, auto const& c) { return g * half_width + c; });

    // Get the ref to local expansion
    auto& locals = cell.locals();

    // Here We take a simple function to test the l2p operator, it just sums all the coordinates of a point and multily by 2.
    auto func = [](auto const&... e) { return 2*(e + ...); };

    // We take the flatten view of our multidimensional grid and apply the function func to the grid.
    // We store it as the local expansion of the cell.
    for(std::size_t n{0}; n<nb_outputs; ++n)
    {
        if constexpr(dimension == 1)
        {
            locals.at(n) = std::apply(func, flatten_views);
        }
        else
        {
            xt::flatten(locals.at(n)) = std::apply(func, flatten_views);
        }
    }

    // we call the operator
    operators::l2p(far_field, cell, leaf);

    auto const& particles = leaf.cparticles();
    auto particle_begin = particles.begin();

    // Now for each outputs in our leaf, we will apply the same function func to the position of the particle
    // and compare this result to the one store in the outputs of the leaf.
    for(std::size_t i{0}; i < nb_particles; ++i)
    {
        // get the particle
        particle_type p(*particle_begin);
        // apply func on the position
        auto result_of_func = std::apply(func, std::array<value_type, dimension>(p.position()));
        // for each output we compare it the result of func(x_i)
        meta::repeat(
          [&result_of_func](auto const& output) {
              REQUIRE(utils::almost_equal(result_of_func, output, std::numeric_limits<value_type>::digits10));
          },
          p.outputs());
        ++particle_begin;
    }
}

TEMPLATE_TEST_CASE("l2p-test-3D-1PV-double", "[l2p-test-3D-1PV-double]"
        , scalfmm::options::chebyshev_<scalfmm::options::dense_>
        , scalfmm::options::uniform_<scalfmm::options::dense_>
        , scalfmm::options::uniform_<scalfmm::options::fft_>
        )
{
    using options_type = TestType;
    using value_type = double;

    SECTION("l2p with d=3, pv=1", "[l2p-uniform,d=3,pv=1]")
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, 3, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

        test_l2p<scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(7);
    }

    SECTION("l2p with d=3, pv=3", "[l2p-uniform,d=3,pv=3]")
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<3>;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, 3, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

        test_l2p<scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(7);
    }
}

TEMPLATE_TEST_CASE("l2p-test-3D-1PV-float", "[l2p-test-3D-1PV-float]"
        , scalfmm::options::chebyshev_<scalfmm::options::dense_>
        , scalfmm::options::uniform_<scalfmm::options::dense_>
        , scalfmm::options::uniform_<scalfmm::options::fft_>
        )
{
    using options_type = TestType;
    using value_type = float;

    SECTION("l2p with d=3, pv=1", "[l2p-uniform,d=3,pv=1]")
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, 3, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

        test_l2p<scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(7);
    }

    SECTION("l2p with d=3, pv=3", "[l2p-uniform,d=3,pv=3]")
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<3>;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, 3, matrix_kernel_type, options_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

        test_l2p<scalfmm::operators::fmm_operators<near_field_type, far_field_type>, value_type>(7);
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    // int result = 0;
    // test_l2p<double, 2, 1>(9);

    return result;
}
