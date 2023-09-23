//
// Units for test fmm
// ----------------------

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/operators/count_kernel/count_kernel.hpp"
//
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"

#include "scalfmm/utils/tensor.hpp"

#include "scalfmm/tools/fma_loader.hpp"

#include <array>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

using namespace scalfmm;

template<std::size_t dim>
inline auto constexpr get_accumulate_shape()
{
    if constexpr(dim == 1)
    {
        return std::array<std::size_t, dim>{1};
    }
    if constexpr(dim == 2)
    {
        return std::array<std::size_t, dim>{1, 1};
    }
    if constexpr(dim == 3)
    {
        return std::array<std::size_t, dim>{1, 1, 1};
    }
    if constexpr(dim == 4)
    {
        return std::array<std::size_t, dim>{1, 1, 1, 1};
    }
}

template<int Dimension>
auto run(const int& tree_height, const int& group_size, bool p2p) -> int
{
    static constexpr std::size_t number_of_physical_values = 1;
    const auto runtime_order = 1;

    // ------------------------------------------------------------------------------
    using particle_type = scalfmm::container::particle<double, Dimension, double, number_of_physical_values, double, 1>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type =
      scalfmm::component::cell<scalfmm::component::grid_storage<double, Dimension, number_of_physical_values, 1>>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box3_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box3_type>;
    //
    // ------------------------------------------------------------------------------
    //
    scalfmm::container::point<double, Dimension> box_center(0.0);
    double box_width{1.};
    //
    container_type* container;
    std::size_t number_of_particles{};
    // generate particles: one par leaf, the octree is full.
    std::cout << "Dimension:   " << Dimension << '\n';
    std::cout << "tree_height: " << tree_height << '\n';
    std::cout << "group_size:  " << group_size << '\n';
    std::cout << "only p2p:    " << std::boolalpha << p2p << '\n';

    double step{box_width / std::pow(2, (tree_height - 1))};
    std::cout << "Step = " << step << '\n';

    auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
    std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';

    xt::xarray<std::tuple<double, double, double, double>> particles(
      std::vector(Dimension, number_of_values_per_dimension));
    number_of_particles = particles.size();

    auto particle_generator = scalfmm::tensor::generate_meshgrid<Dimension>(xt::linspace(
      double(-box_width / 2.) + step * 0.5, double(box_width / 2.) - step * 0.5, number_of_values_per_dimension));
    auto eval_generator = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::eval(std::forward<decltype(xs)>(xs))...); }, particle_generator);
    auto flatten_views = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::flatten(std::forward<decltype(xs)>(xs))...); }, eval_generator);

    auto particle_flatten_views = xt::flatten(particles);
    container = new container_type(particles.size());
    auto container_it = std::begin(*container);
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        *container_it = std::apply(
          [&i](auto&&... xs) { return std::make_tuple(std::forward<decltype(xs)>(xs)[i]..., 0., 0.); }, flatten_views);
        ++container_it;
    }

    box3_type box3(box_width, box_center);

    group_tree_type gtree(static_cast<std::size_t>(tree_height), runtime_order, box3,
                          static_cast<std::size_t>(group_size), static_cast<std::size_t>(group_size), *container);

    //  using fmm_operator_type = count_kernels::particles::count_fmm_operator<Dimension>;
    using fmm_operator_type = operators::fmm_operators<count_kernels::particles::count_near_field,
                                                       count_kernels::particles::count_far_field<Dimension>>;
    bool mutual = false;

    count_kernels::particles::count_near_field nf(mutual);
    count_kernels::particles::count_far_field<Dimension> ff{};
    fmm_operator_type fmm_operator(nf, ff);
    auto operator_to_proceed = p2p ? scalfmm::algorithms::p2p : scalfmm::algorithms::all;
#ifdef COUNT_USE_OPENMP
    scalfmm::algorithms::omp::task_dep(gtree, fmm_operator, operator_to_proceed);
#else
    scalfmm::algorithms::sequential::sequential(gtree, fmm_operator, operator_to_proceed);

#endif
    std::size_t nb_particles_min = 2 * number_of_particles, nb_particles_max = 0;
    bool right_number_of_particles = true;
    if(!p2p)
    {   // full FMM
        size_t nb_part{};
        scalfmm::component::for_each_leaf(std::begin(gtree), std::end(gtree),
                                          [&right_number_of_particles, &nb_part, number_of_particles, &nb_particles_max,
                                           &nb_particles_min](auto const& leaf)
                                          {
                                              auto p_r = typename leaf_type::proxy_type(*(leaf.begin()));
                                              nb_part = p_r.outputs()[0];
                                              nb_particles_max = std::max(nb_particles_max, nb_part);
                                              nb_particles_min = std::min(nb_particles_min, nb_part);
                                              if(nb_part != number_of_particles)
                                              {
                                                  std::cout << cpp_tools::colors::red
                                                            << "wrong number of particles - index " << leaf.index()
                                                            << " nb particles " << nb_part << std::endl;
                                                  right_number_of_particles = false;
                                              }
                                          });
        std::cout << cpp_tools::colors::reset << '\n';

        if(right_number_of_particles)
        {
            std::cout << "Found the right number of particles - nb particles " << number_of_particles << std::endl;
        }
        else
        {
            std::cout << "wrong number of particles - nb particles (min) " << nb_particles_min << "  (max) "
                      << nb_particles_max << " (expected) " << number_of_particles << std::endl;
        }
        REQUIRE(right_number_of_particles);
    }
    else
    {
        std::vector<std::size_t> right_number_of_particles_p2p{4, 6, 6, 9, 6, 4, 9, 6, 6, 9, 4, 6, 9, 6, 6, 4};
        std::size_t i{0};
        scalfmm::component::for_each_leaf(
          std::begin(gtree), std::end(gtree),
          [&right_number_of_particles_p2p, &right_number_of_particles, &i](auto const& leaf)
          {
              size_t nb_part = std::get<0>(*scalfmm::container::outputs_begin(leaf.particles()));
              if(nb_part != right_number_of_particles_p2p.at(i++))
              {
                  std::cout << cpp_tools::colors::red << "wrong number of particles - index " << leaf.index()
                            << " nb particles " << nb_part << std::endl;
                  right_number_of_particles = false;
              }
          });
        std::cout << cpp_tools::colors::reset << '\n';
        REQUIRE(right_number_of_particles);
    }
    return right_number_of_particles;
}

TEMPLATE_TEST_CASE("test count 1d", "[test-count-1d]", double)
{
    SECTION("count 1d", "[count1d]") { run<1>(5, 10, false); }   // h = 5
}
#ifndef NO_DEBUG_OO
TEMPLATE_TEST_CASE("test count 2d", "[test-count-2d]", double)
{
    SECTION("count 2d", "[count2d]") { run<2>(5, 10, false); }
}
TEMPLATE_TEST_CASE("test count 3d", "[test-count-3d]", double)
{
    SECTION("count 3d", "[count3d]") { run<3>(5, 10, false); }
}

TEMPLATE_TEST_CASE("test count 4d", "[test-count-4d]", double)
{
    SECTION("count 4d", "[count4d]") { run<4>(5, 10, false); }
}

TEMPLATE_TEST_CASE("test count p2p", "[test-count-p2p]", double)
{
    SECTION("count p2p", "[count-p2p]") { run<2>(3, 2, true); }
}
#endif
int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    // int result = 0;
    // test_l2p<double, 2, 1>(9);

    return result;
}
