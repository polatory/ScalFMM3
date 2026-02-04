/**
 * @file check_periodic.cpp
 * @author Olivier Coulaud (olivier.coulaud@inria.fr)
 * @brief
 *
 * @version 0.1
 * @date 2022-02-07
 *
 * @copyright Copyright (c) 2022
 *
 */
// @FUSE_FFTW
// @FUSE_CBLAS

// make check_periodic
//  ./examples/Release/check_periodic --input-file check_per.fma --d 2 --per 1,1 -o 4 --tree-height 4 -gs 2 --check
// #include <algorithm>
// #include <array>
#include <string>
// #include <tuple>
// #include <utility>
// #include <vector>
// Tools
#include <cpp_tools/colors/colorized.hpp>
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
// Local files
// Scalfmm includes
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/tools/fma_loader.hpp"
// Tree
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
// replicated container
#include "scalfmm/meta/utils.hpp"
// Fmm operators
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/periodicity.hpp"
#include "units_fmm.hpp"

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

template<typename Container>
auto read_data(const std::string& filename)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using value_type = typename particle_type::position_value_type;
    static constexpr std::size_t dimension{particle_type::dimension};
    const bool verbose{false};

    scalfmm::io::FFmaGenericLoader<value_type, dimension> loader(filename, verbose);

    const std::size_t number_of_particles{loader.getNumberOfParticles()};
    // std::cout << cpp_tools::colors::yellow << "[file][n_particles] : " << number_of_particles
    //           << cpp_tools::colors::reset << '\n';
    const auto width{loader.getBoxWidth()};
    // std::cout << cpp_tools::colors::yellow << "[file][box_width] : " << width << cpp_tools::colors::reset << '\n';
    const auto center{loader.getBoxCenter()};
    // std::cout << cpp_tools::colors::yellow << "[file][box_centre] : " << center << cpp_tools::colors::reset << '\n';

    auto nb_val_to_red_per_part = loader.getNbRecordPerline();
    // could be a problem for binary file (float double)
    std::vector<value_type> values_to_read(nb_val_to_red_per_part);

    container_type container(number_of_particles);

    for(std::size_t idx = 0; idx < number_of_particles; ++idx)
    {
        loader.fillParticle(values_to_read.data(), nb_val_to_red_per_part);
        particle_type p;
        std::size_t ii{0};
        for(auto& e: p.position())
        {
            e = values_to_read[ii++];
        }
        for(auto& e: p.inputs())
        {
            e = values_to_read[ii++];
        }
        // p.variables(values_to_read[ii++], idx, 1);
        p.variables(idx, 1);
        container.insert_particle(idx, p);
    }
    return std::make_tuple(container, center, width);
}
template<typename Tree, typename Container>
auto check_output_per(Container const& part, Tree const& tree) -> scalfmm::utils::accurater<double>
{
    scalfmm::utils::accurater<double> error;

    auto nb_out = part[0].sizeof_outputs();
    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&nb_out, &part, &error](auto& leaf)
                                      {
                                          for(auto const p_tuple_ref: leaf)
                                          {
                                              // We construct a particle type for classical acces
                                              const auto& p = typename Tree::leaf_type::const_proxy_type(p_tuple_ref);

                                              const auto& idx = std::get<0>(p.variables());

                                              auto output = p.outputs();
                                              auto output_ref = part[idx].outputs();

                                              for(int n{0}; n < nb_out; ++n)
                                              {
                                                  //   std::cout << i << " " << n << " " << output_ref.at(n) << "  "
                                                  //             << output.at(n) << std::endl;
                                                  error.add(output_ref.at(n), output.at(n));
                                              }
                                          }
                                      });
    return error;
}
template<int Dimension, typename value_type, class FMM_OPERATOR_TYPE, typename Array>
auto test_periodic(const int& tree_height, const int& group_size, const std::size_t order, Array const& pbc,
                   const int& nb_level_above_root, const std::string& input_file) -> int

{
    using near_matrix_kernel_type = typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;

    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};

    // position
    // number of input values and type
    // number of output values
    // variables: original index, original box (1) otherwise 0 for the replicated boxes
    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, std::size_t, int>;
    using container_type = scalfmm::container::particle_container<particle_type>;

    using position_type = typename particle_type::position_type;

    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    //

    using near_matrix_kernel_type = typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;
    using far_field_type = typename FMM_OPERATOR_TYPE::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;

    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;

    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;
    ///////////////////////////////////////////////////////////////
    // Particles read
    position_type box_center{};
    value_type box_width{};
    container_type container{};
    std::tie(container, box_center, box_width) = read_data<container_type>(input_file);

    //  read_data<Dimension>(input_file, container, box_center, box_width);
    //
    ////////////////////////////////////////////////////
    // Build tree
    box_type box(box_width, box_center);
    box.set_periodicity(pbc);
    // std::cout << "Box:          " << box << std::endl;
    auto extended_width{box.extended_width(nb_level_above_root)};
    auto extended_center{box.extended_center(nb_level_above_root)};
    box_type ext_box(extended_width, extended_center);
    ext_box.set_periodicity(pbc);

    std::cout << "extended Box: " << ext_box << std::endl;

    group_tree_type tree(tree_height, order, box, group_size, group_size, container);
    tree.set_levels_above_root(nb_level_above_root);
    //
    ///////////////////////////////////
    // auto operator_to_proceed = scalfmm::algorithms::farfield;
    auto operator_to_proceed = scalfmm::algorithms::all;
    //
    ////////////////////////////////////////////////////

    ///////////////////////////////////////////////////
    // Periodic run
    far_matrix_kernel_type mk_far{};
    auto total_height = tree_height /*+ nb_level_above_root + 2*/;
    interpolator_type interpolator(mk_far, order, total_height, extended_width);
    // interpolator_type interpolator(mk_far, order, static_cast<std::size_t>(tree_height), box.width(0));
    typename FMM_OPERATOR_TYPE::far_field_type far_field(interpolator);
    //
    near_matrix_kernel_type mk_near{};
    typename FMM_OPERATOR_TYPE::near_field_type near_field(mk_near);

    std::cout << "Kernel (near): " << mk_near.name() << std::endl;
    std::cout << "Kernel (far):  " << mk_far.name() << std::endl;
    FMM_OPERATOR_TYPE fmm_operator(near_field, far_field);
    auto neighbour_separation = fmm_operator.near_field().separation_criterion();
    const bool mutual = true;
    //
    tree.build_interaction_lists(tree, neighbour_separation, mutual);
    scalfmm::algorithms::sequential::sequential(tree, tree, fmm_operator, operator_to_proceed);
    // std::clog << "END algorithm\n";

    ///////////////////////////////////////////////////
    // non periodic run
    // generate replicated distribution
    // using particle_rep_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near,
    // value_type, nb_outputs_near, std::size_t, int>; using container_rep_type =
    // scalfmm::container::particle_container<particle_type>; using leaf_rep_type =
    // scalfmm::component::leaf<particle_rep_type>; using group_tree2_type =
    // scalfmm::component::group_tree<cell_type, leaf_rep_type, box_type>;

    auto replicated_container = scalfmm::utils::replicated_distribution_grid_3x3(container, box_width, pbc);
    std::cout << "Kernel (direct): " << mk_near.name() << std::endl;
    scalfmm::algorithms::full_direct(replicated_container, mk_near);

    const int ref = 1;

    auto extracted_cont = scalfmm::utils::extract_particles_from_ref(replicated_container, container.size(), ref);
    // auto error1 = check_output(extracted_cont, tree);
    // std::cout << error1 << std::endl;
    value_type eps = std::pow(10.0, 1 - order);
    auto error{check_output_per(extracted_cont, tree).get_relative_l2_norm()};
    bool works = error < eps;
    std::cout << cpp_tools::colors::magenta << "Error " << error << '\n' << cpp_tools::colors::reset;
    if(works)
    {
        std::cout << cpp_tools::colors::blue << " Test Ok \n" << cpp_tools::colors::reset;
    }
    else
    {
        std::cout << cpp_tools::colors::red << " Test is WRONG !! the error must be around  " << eps << " \n "
                  << cpp_tools::colors::reset;
    }

    return works;
}

TEMPLATE_TEST_CASE("periodic-2D-full-double",
                   "[periodic-2D-full-double]"
                   //  , scalfmm::options::chebyshev_<scalfmm::options::dense_>
                   ,
                   float, double
                   // , scalfmm::options::uniform_<scalfmm::options::fft_>
)
{
    static constexpr std::size_t dimension{2};

    using value_type = TestType;
    std::string path{TEST_DATA_FILES_PATH};
    std::vector<bool> pbc(dimension, true);

    SECTION("periodic with d=2, pbc=true,true", "[periodic-uniform,d=2, pbc=true]")
    {
        std::cout << "value_type = double\n";
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
        //
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type =
          interpolator_alias<value_type, dimension, matrix_kernel_type, scalfmm::options::chebyshev_<>>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        path.append("unitCube_2d_100_per.fma");

        REQUIRE(test_periodic<dimension, value_type, fmm_operators_type>(4, 2, 4, pbc, 0, path));
    }

    SECTION("periodic with d=2, pbc=true,false", "[periodic-uniform,d=2, pbc=true,false]")
    {
        std::cout << "value_type = double\n";

        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;

        //
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type =
          interpolator_alias<value_type, dimension, matrix_kernel_type, scalfmm::options::chebyshev_<>>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        path.append("unitCube_2d_100_per.fma");
        pbc[0] = false;

        REQUIRE(test_periodic<dimension, value_type, fmm_operators_type>(4, 2, 4, pbc, 0, path));
    }
    SECTION("periodic with d=2, pbc=true,false", "[periodic-uniform,d=2, pbc=true,false]")
    {
        std::cout << "value_type = double\n";

        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;

        //
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type =
          interpolator_alias<value_type, dimension, matrix_kernel_type, scalfmm::options::chebyshev_<>>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        path.append("unitCube_2d_100_per.fma");
        pbc[0] = true;
        pbc[1] = false;

        REQUIRE(test_periodic<dimension, value_type, fmm_operators_type>(4, 2, 4, pbc, 0, path));
    }
}
int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}