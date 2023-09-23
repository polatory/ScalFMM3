//
// Units for test fmm
// ----------------------
#include <array>
#include <iostream>
#include <string>
#include <tuple>
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/count_kernel/count_kernel.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/tree_io.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/io.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/source_target.hpp"
#include "scalfmm/utils/tensor.hpp"

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

using namespace scalfmm;
using value_type = double;

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
    const auto runtime_order = 1;

    constexpr int zeros{1};
    static constexpr std::size_t nb_inputs{1};
    static constexpr std::size_t nb_outputs{1};
    static constexpr std::size_t number_of_physical_values = 1;
    const std::size_t dimpow2 = scalfmm::math::pow(2, Dimension);

    // ------------------------------------------------------------------------------

    using point_type = scalfmm::container::point<value_type, Dimension>;
    using particle_source_type =
      scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs, value_type, zeros, std::size_t>;
    using particle_target_type =
      scalfmm::container::particle<value_type, Dimension, value_type, zeros, value_type, nb_outputs, std::size_t>;

    using cell_type =
      scalfmm::component::cell<scalfmm::component::grid_storage<value_type, Dimension, number_of_physical_values, 1>>;

    //
    ///////////////////////////////////////////////////////////////////////////

    // Construct the container of particles
    // using container_type = scalfmm::container::particle_container<particle_source_type>;
    using box_type = scalfmm::component::box<point_type>;
    //
    using leaf_source_type = scalfmm::component::leaf_view<particle_source_type>;
    using leaf_target_type = scalfmm::component::leaf_view<particle_target_type>;

    using cell_type =
      scalfmm::component::cell<scalfmm::component::grid_storage<value_type, Dimension, number_of_physical_values, 1>>;

    using tree_source_type = scalfmm::component::group_tree_view<cell_type, leaf_source_type, box_type>;
    using tree_target_type = scalfmm::component::group_tree_view<cell_type, leaf_target_type, box_type>;
    //
    //
    // ------------------------------------------------------------------------------
    //
    point_type box_center(0.0);
    value_type box_width{1.0};

    std::vector<particle_source_type> particles_source;
    std::vector<particle_target_type> particles_target;

    box_type box(box_width, box_center);
    //
    // generate particles: one par leaf, the octree is full.
    std::cout << "Dimension:   " << Dimension << '\n';
    std::cout << "tree_height: " << tree_height << '\n';
    std::cout << "group_size:  " << group_size << '\n';
    std::cout << "only p2p:    " << std::boolalpha << p2p << '\n';

    auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
    std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';
    // generate particles: one par leaf, the octree is full.
    auto number_of_particles = std::pow(dimpow2, (tree_height - 1));
    std::cout << "number_of_particles = " << number_of_particles << " box_width " << box_width << '\n';

    particles_source.resize(number_of_particles);
    particles_target.resize(number_of_particles);

    // step is the leaf size
    double step{box_width / std::pow(2, (tree_height - 1))};
    std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';
    std::cout << "Step = " << step << '\n';
    // start is the center of the first box
    auto start = -box_width * 0.5 + step * 0.5;
    auto delta = step * 0.25;   // used to separate source and target

    for(std::size_t index{0}; index < number_of_particles; ++index)
    {
        // get coord of the cell in the grid with the morton index
        auto coord = scalfmm::index::get_coordinate_from_morton_index<Dimension>(index);

        point_type pos{coord};
        particle_source_type p_source;
        std::size_t ii{0};
        for(auto& e: p_source.position())
        {
            e = start + step * pos[ii++] - delta;
        }
        particles_source[index] = p_source;
        //
        particle_target_type p_target;
        ii = 0;
        for(auto& e: p_target.position())
        {
            e = start + step * pos[ii++] + delta;
        }
        particles_target[index] = p_target;
    }

    tree_source_type tree_source(static_cast<std::size_t>(tree_height), runtime_order, box,
                                 static_cast<std::size_t>(group_size), static_cast<std::size_t>(group_size),
                                 particles_source);

    tree_target_type tree_target(static_cast<std::size_t>(tree_height), runtime_order, box,
                                 static_cast<std::size_t>(group_size), static_cast<std::size_t>(group_size),
                                 particles_target);

    //  using fmm_operator_type = count_kernels::particles::count_fmm_operator<Dimension>;
    using fmm_operator_type = operators::fmm_operators<count_kernels::particles::count_near_field,
                                                       count_kernels::particles::count_far_field<Dimension>>;
    bool mutual = false;
    count_kernels::particles::count_near_field nf(mutual);
    count_kernels::particles::count_far_field<Dimension> ff{};
    fmm_operator_type fmm_operator(nf, ff);
    auto operator_to_proceed = p2p ? scalfmm::algorithms::p2p : scalfmm::algorithms::all;
    //
#ifdef COUNT_USE_OPENMP
    scalfmm::algorithms::omp::task_dep(tree_source, tree_target, fmm_operator, operator_to_proceed);
#else
    scalfmm::algorithms::sequential::sequential(tree_source, tree_target, fmm_operator, operator_to_proceed);
#endif
    std::size_t nb_particles_min = 2 * number_of_particles, nb_particles_max = 0;
    bool right_number_of_particles = true;
    if(!p2p)
    {   // full FMM
        size_t nb_part{};
        scalfmm::component::for_each_leaf(std::begin(tree_target), std::end(tree_target),
                                          [&right_number_of_particles, &nb_part, number_of_particles, &nb_particles_max,
                                           &nb_particles_min](auto const& leaf)
                                          {
                                              auto p_r = typename leaf_target_type::proxy_type(*(leaf.begin()));
                                              nb_part = p_r.outputs()[0];
                                              nb_particles_max = std::max(nb_particles_max, nb_part);
                                              nb_particles_min = std::min(nb_particles_min, nb_part);
                                              if(nb_part != number_of_particles)
                                              {
                                                  std::cout << cpp_tools::colors::red
                                                            << "wrong number of particles - index " << leaf.index()
                                                            << " nb particles " << nb_part << " Difference "
                                                            << number_of_particles - nb_part << std::endl;
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
            std::cout << " Display dependencies\n";
            scalfmm::io::trace_group_dependencies(std::cout, tree_source, tree_target);
            std::cout << cpp_tools::colors::reset;
            std::cout << cpp_tools::colors::reset;
            std::string outputFile{"random3_target.fma"};
            std::cout << "Write targets in " << outputFile << std::endl;
            scalfmm::io::FFmaGenericWriter<double> writer_t(outputFile);
            writer_t.writeDataFromTree(tree_target, number_of_particles);

            outputFile = "random3_source.fma";
            scalfmm::io::FFmaGenericWriter<double> writer_s(outputFile);
            writer_s.writeDataFromTree(tree_source, number_of_particles);
            // std::string outName("output_tree_source");
            // scalfmm::tools::io::save(outName, tree_source, "coucou");
            // std::string outName1("output_tree_target");
            // scalfmm::tools::io::save(outName1, tree_target, "coucou");
        }
        REQUIRE(right_number_of_particles);
    }
    else
    {
        std::vector<std::size_t> right_number_of_particles_p2p;
        if(Dimension == 1)
        {
            right_number_of_particles_p2p = std::vector<std::size_t>{2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2};
        }
        else if(Dimension == 2)
        {
            right_number_of_particles_p2p = std::vector<std::size_t>{4, 6, 6, 9, 6, 4, 9, 6, 6, 9, 4, 6, 9, 6, 6, 4};
        }
        //      std::vector<std::size_t> right_number_of_particles_p2p
        // if(Dimension ==2
        // std::vector<std::size_t> right_number_of_particles_p2p{4, 6, 6, 9, 6, 4, 9, 6, 6, 9, 4, 6, 9, 6, 6, 4};
        // std::vector<std::size_t>{4, 6, 6, 9, 6, 4, 9, 6, 6, 9, 4, 6, 9, 6, 6, 4};
        std::size_t i{0};
        scalfmm::component::for_each_leaf(
          std::begin(tree_target), std::end(tree_target),
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
        // if(not right_number_of_particles)
        // {
        //     std::cout << " Display dependencies\n" << cpp_tools::colors::blue << " Tree source: \n";
        //     scalfmm::io::trace_group_dependencies(std::cout, tree_source);
        //     std::cout << cpp_tools::colors::red << "\n\n Tree target: \n";
        //     scalfmm::io::trace_group_dependencies(std::cout, tree_target);
        //     std::cout << cpp_tools::colors::reset;
        // }
        REQUIRE(right_number_of_particles);
    }
    return right_number_of_particles;
}
#ifdef _OC_TT
TEMPLATE_TEST_CASE("test count 1d", "[test-count-1d]", double)
{
    SECTION("count 1d", "[count1d]") { run<1>(5, 5, false); }   // h = 5
}
#endif
TEMPLATE_TEST_CASE("test count 2d", "[test-count-2d]", double)
{
    SECTION("count 2d", "[count2d]") { run<2>(5, 10, false); }
}
#ifdef _OC_TT
TEMPLATE_TEST_CASE("test count 2d", "[test-count-2d]", float)
{
    SECTION("count 2d", "[count2d]") { run<2>(5, 10, false); }
}
TEMPLATE_TEST_CASE("test count 3d", "[test-count-3d]", double)
{
    SECTION("count 3d", "[count3d]") { run<3>(2, 10, false); }
}

// TEMPLATE_TEST_CASE("test count 4d", "[test-count-4d]", double)
// {
//     SECTION("count 4d", "[count4d]") { run<4>(5, 10, false); }
// }

TEMPLATE_TEST_CASE("test count 1d-p2p", "[test-count-1d-p2p]", double)
{
    SECTION("count 1d p2p", "[count1d-p2p]") { run<1>(5, 10, true); }   // h = 5
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
