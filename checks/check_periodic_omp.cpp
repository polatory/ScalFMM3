/**
 * @file check_periodic_omp.cpp
 * @author Olivier Coulaud (olivier.coulaud@inria.fr)
 * @brief
 *
 * @version 0.1
 * @date 2022-02-07
 *
 * @copyright Copyright (c) 2022
 *
 */

// make check_periodic
// ./check/Release/check_periodic_omp --input-file ../data/units/check_per.fma --d 2 --per 1,1 -o 5 --tree-height 4
//  -gs 2 --check
//
#include <algorithm>
#include <array>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
// Tools
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
// parameters
#include "scalfmm/utils/parameters.hpp"
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
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/options/options.hpp"

#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/periodicity.hpp"

namespace local_args
{

    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n   0 for dimension 2, 1 for dimension 3";
        using type = int;
    };

    struct check
    {
        cpp_tools::cl_parser::str_vec flags = {"--check"};
        std::string description = "Check with p2p ";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
    struct displayParticles
    {
        cpp_tools::cl_parser::str_vec flags = {"--display-particles", "-dp"};
        std::string description = "Display all particles  ";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
    struct displayCells
    {
        cpp_tools::cl_parser::str_vec flags = {"--display-cells", "-dc"};
        std::string description = "Display the cells at leaf level  ";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
}   // namespace local_args

template<typename Container>
auto read_data(const std::string& filename)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using value_type = typename particle_type::position_value_type;
    static constexpr std::size_t dimension{particle_type::dimension};
    const bool verbose{true};

    scalfmm::io::FFmaGenericLoader<value_type, dimension> loader(filename, verbose);

    const std::size_t number_of_particles{loader.getNumberOfParticles()};
    std::cout << cpp_tools::colors::yellow << "[file][n_particles] : " << number_of_particles
              << cpp_tools::colors::reset << '\n';
    const auto width{loader.getBoxWidth()};
    std::cout << cpp_tools::colors::yellow << "[file][box_width] : " << width << cpp_tools::colors::reset << '\n';
    const auto center{loader.getBoxCenter()};
    std::cout << cpp_tools::colors::yellow << "[file][box_centre] : " << center << cpp_tools::colors::reset << '\n';

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
auto check_output(Container const& part, Tree const& tree) -> scalfmm::utils::accurater<double>
{
    scalfmm::utils::accurater<double> error;

    auto nb_out = part[0].sizeof_outputs();
    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&nb_out, &part, &error](auto& leaf)
                                      {
                                          for(auto const p_tuple_ref: leaf)
                                          {
                                              // We construct a particle type for classical acces
                                              const auto p = typename Tree::leaf_type::const_proxy_type(p_tuple_ref);
                                              //     auto p = part.position();
                                              // }
                                              //     const auto& container = leaf.cparticles();
                                              //     const auto nb_elt = std::distance(container.begin(),
                                              //     container.end()); for(std::size_t i = 0; i < nb_elt; ++i)
                                              //     {
                                              //   const auto& p = container.particle(i);
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
auto run(const int& tree_height, const int& group_size, const std::size_t order, Array const& pbc,
         const int& nb_level_above_root, const std::string& input_file, const std::string& output_file,
         const bool check, const bool displayCells, const bool displayParticles) -> int

{
    using near_matrix_kernel_type = typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;

    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};

    // position
    // number of input values and type
    // number of output values
    // variables original index, original box (1) otherwise 0 for the replicated boxes
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
    std::cout << "nb_level_above_root: " << nb_level_above_root << std::endl;
    std::cout << "Box:          " << box << std::endl;
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
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](tree, fmm_operator, operator_to_proceed);

    std::clog << "END algorithm\n";
    if(check)
    {
        ///////////////////////////////////////////////////
        // non periodic run
        // generate replicated distribution

        cpp_tools::timers::timer time{};

        std::cout << cpp_tools::colors::green << "[status] creating replicated particles distribution...\n"
                  << cpp_tools::colors::reset;
        time.tic();
        auto replicated_container = scalfmm::utils::replicated_distribution_grid_3x3(container, box_width, pbc);
        std::cout << cpp_tools::colors::green << "[status] ...done.\n" << cpp_tools::colors::reset;
        std::cout << cpp_tools::colors::magenta
                  << "[timings][constructing replicated distribution] = " << time.elapsed() << "ms\n"
                  << cpp_tools::colors::reset;
        std::cout << "Kernel (direct): " << mk_near.name() << std::endl;
        scalfmm::algorithms::full_direct(replicated_container, mk_near);

        const int ref = 1;

        // auto extracted_cont = extract(tree_replicated, container.size(), ref);
        // auto error = check_output(extracted_cont, tree) ;
        // std::cout << error<<std::endl;
        auto extracted_cont = scalfmm::utils::extract_particles_from_ref(replicated_container, container.size(), ref);
        auto error1 = check_output(extracted_cont, tree);
        std::cout << error1 << std::endl;
    }
    else
    {
        if(displayCells)
        {
            for_each_cell(std::begin(tree), std::end(tree), tree.height() - 1,
                          [](auto& cell)
                          {
                              scalfmm::utils::print_cell(cell);
                              std::cout << std::endl;
                          });
        }
        value_type energy{0}, total_charge{0};
        auto print_result_to_scalfmm2 = [](auto& leaf)
        {
            int i{0};
            auto inputs_iterator = scalfmm::container::inputs_begin(leaf.cparticles());
            auto outputs_begin = scalfmm::container::outputs_begin(leaf.particles());
            auto outputs_end = scalfmm::container::outputs_end(leaf.particles());
            // In ScalFMM2 the forces are multiplied by  q
            auto inputs_begin_lazy = scalfmm::container::inputs_begin(leaf.particles());
            // we dereference to evaluate the lazy pointer
            auto inputs_begin = *inputs_begin_lazy;
            //  You get the first input value and you take
            // its address in order to increment if
            auto q = &std::get<0>(inputs_begin);
            // construct a sequence to access directly to the force in the output
            using range_force = scalfmm::meta::make_range_sequence<1, particle_type::outputs_size>;
            // the outputs are [ p, fx,fy, fz] and we construct [ p, q*fx,q*fy, q*fz]
            // where q is the first input of teh particle
            for(auto it = outputs_begin; it != outputs_end; ++it, ++q)
            {
                // out =[ p, fx,fy, fz]
                scalfmm::meta::repeat([q](auto& v) { v *= *q; }, scalfmm::meta::sub_tuple(*it, range_force{}));
            }
        };
        auto computeEnergy = [&energy, &total_charge](auto& leaf)
        {
            for(auto const p_tuple_ref: leaf)
            {
                // We construct a particle type for classical acces
                const auto proxy = typename leaf_type::const_proxy_type(p_tuple_ref);
                // get charge
                auto q = proxy.inputs();
                // get forces
                auto& out = proxy.outputs();
                // to check with scalfmm 2
                energy += q[0] * out[0];
                total_charge += q[0];
            }
            // auto& particles{leaf.particles()};
            // for(std::size_t i{0}; i < leaf.size(); ++i)
            // {
            //     auto proxy = particles.at(i);
            //     // get charge
            //     auto q = proxy.inputs();
            //     // get forces
            //     auto& out = proxy.outputs();
            //     // to check with scalfmm 2
            //     energy += q[0] * out[0];
            //     total_charge += q[0];

            //     // }
            // }
        };
        if(displayParticles)
        {
            for_each_leaf(std::begin(tree), std::end(tree), print_result_to_scalfmm2);
        }
        for_each_leaf(std::begin(tree), std::end(tree), computeEnergy);

        std::cout << "Energy: " << energy << " TotalPhysicalValue: " << total_charge << std::endl;
    }
    return 1;
}
// using option = scalfmm::options::chebyshev_<>;
using option = scalfmm::options::uniform_<scalfmm::options::fft_>;

template<typename V, std::size_t D, typename MK>
using interpolator_alias = scalfmm::interpolation::interpolator<V, D, MK, option>;

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    static constexpr std::size_t dimension{2};
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, args::input_file(), args::output_file(), args::tree_height{}, args::block_size{},
      args::order{}, args::thread_count{},
      //, args::log_file{}, args::log_level{}
      args::pbc{}, args::extended_tree_height{},   // periodicity
      local_args::dimension{}, local_args::check{}, local_args::displayParticles{}, local_args::displayCells{}

    );

    parser.parse(argc, argv);
    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';

    const int group_size{parser.get<args::block_size>()};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';

    const std::string input_file{parser.get<args::input_file>()};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                  << '\n';
    }

    const std::string output_file{parser.get<args::output_file>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    const auto order{parser.get<args::order>()};
    std::vector<bool> pbc(dimension, false);
    auto nb_level_above_root = parser.get<args::extended_tree_height>();

    if(parser.exists<args::pbc>())
    {
        pbc = parser.get<args::pbc>();
    }
    std::cout << "pbc: " << std::boolalpha;
    for(auto e: pbc)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;
    bool check{parser.exists<local_args::check>()};
    bool displayCells{parser.exists<local_args::displayCells>()};
    bool displayParticles{parser.exists<local_args::displayParticles>()};
    //
    const std::size_t nb_threads{parser.get<args::thread_count>()};
    omp_set_dynamic(0);
    omp_set_num_threads(nb_threads);
    //
//#define ONE_OVER_R
#ifdef ONE_OVER_R
    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
#else
#define GRAD_ONE_OVER_R
#ifdef GRAD_ONE_OVER_R
    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
    using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
#else
    // VAL_GRAD_ONE_OVER_R
    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
    using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;

#endif
#endif

    using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
    //
    using value_type = double;
    using interpolation_type = interpolator_alias<value_type, dimension, far_matrix_kernel_type>;
    using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

    using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    run<dimension, value_type, fmm_operators_type>(tree_height, group_size, order, pbc, nb_level_above_root, input_file,
                                                   output_file, check, displayCells, displayParticles);
}
