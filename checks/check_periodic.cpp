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

// make check_periodic
//  ./check/Release/check_periodic --input-file ../data/units/check_per.fma --d 2 --per 1,1 -o 4 --tree-height 4
//  -gs 2 --check
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
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/periodicity.hpp"

namespace local_args
{

    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n\t0 for dimension 2, 1 for dimension 3";
        using type = int;
    };

    struct kernel : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--kernel", "--k"};
        std::string description = "Kernel : \n\t0 for 'one_over_r' 2, 1 for 'grad_one_over_r<dim>, 2 for "
                                  "'val_grad_one_over_r<dim>, (default) 'one_over_r'";
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

                                              const auto& idx = std::get<0>(p.variables());

                                              auto output = p.outputs();
                                              auto output_ref = part[idx].outputs();
                                              for(int n{0}; n < nb_out; ++n)
                                              {
                                                  error.add(output_ref.at(n), output.at(n));
                                              }
                                          }
                                          //   const auto& container = leaf.cparticles();
                                          //   const auto nb_elt = std::distance(container.begin(), container.end());
                                          //   for(std::size_t i = 0; i < nb_elt; ++i)
                                          //   {
                                          //       const auto& p = container.particle(i);
                                          //       const auto& idx = std::get<0>(p.variables());

                                          //       auto output = p.outputs();
                                          //       auto output_ref = part[idx].outputs();
                                          //       for(int n{0}; n < nb_out; ++n)
                                          //       {
                                          //           //   std::cout << i << " " << n << " " << output_ref.at(n) << " "
                                          //           //             << output.at(n) << std::endl;
                                          //           error.add(output_ref.at(n), output.at(n));
                                          //       }
                                          //   }
                                      });
    return error;
}

template<int Dimension, typename value_type, class FMM_OPERATOR_TYPE, typename Array>
auto run(const int& tree_height, const int& group_size, const std::size_t order, Array const& pbc,
         const int& nb_level_above_root, const std::string& input_file, const std::string& output_file,
         const bool check, const bool displayCells, const bool displayParticles) -> void

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

    // TEMP AG
    auto const& interp_tensor{interpolator.cinterpolator_tensor()};
    auto shape = interp_tensor.shape();
    std::cout << "interp tensor shape : " << std::endl;
    std::cout << "\t" << shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << std::endl;

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
    scalfmm::algorithms::sequential::sequential(tree, fmm_operator, operator_to_proceed);
    std::clog << "END algorithm\n";
    if(check)
    {
        ///////////////////////////////////////////////////
        // non periodic run

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
                              scalfmm::io::print_cell(cell);
                              std::cout << std::endl;
                          });
        }
        value_type energy{0}, total_charge{0};
        auto print_result_to_scalfmm2 = [](auto& leaf)
        {
            for(auto const p_tuple_ref: leaf)
            {
                // We construct a particle type for classical acces
                auto proxy = typename leaf_type::proxy_type(p_tuple_ref);
                auto const& q = proxy.inputs();
                // get forces
                auto& out = proxy.outputs();
                // In ScalFMM2 the forces are multiplied by  q
                int start = near_matrix_kernel_type::kn == Dimension + 1 ? 1 : 0;
                for(int j = start; j < near_matrix_kernel_type::kn; ++j)
                {
                    out[j] *= q[0];
                }
                // std::cout << i << " p " << proxy << " " << std::endl;
            }
        };
        auto computeEnergy = [&energy, &total_charge](auto& leaf)
        {
            for(auto const p_tuple_ref: leaf)
            {
                // We construct a particle type for classical acces
                auto proxy = typename leaf_type::proxy_type(p_tuple_ref);
                auto const& q = proxy.inputs();
                // get forces
                auto& out = proxy.outputs();
                // In ScalFMM2 the forces are multiplied by  q
                energy += q[0] * out[0];
                total_charge += q[0];
                // std::cout << i << " p " << proxy << " " << std::endl;
            }
        };
        if(displayParticles)
        {
            for_each_leaf(std::begin(tree), std::end(tree), print_result_to_scalfmm2);
        }
        for_each_leaf(std::begin(tree), std::end(tree), computeEnergy);

        std::cout << "Energy: " << energy << " TotalPhysicalValue: " << total_charge << std::endl;
    }
}

/// @brief
/// @tparam V
/// @tparam MK
/// @tparam D
template<typename V, std::size_t D, typename MK>
using interpolator_alias = scalfmm::interpolation::interpolator<V, D, MK, scalfmm::options::chebyshev_<>>;

/// @brief
/// @tparam ValueType
/// @tparam Dimension
/// @param tree_height
/// @param group_size
/// @param order
/// @param pbc
/// @param nb_level_above_root
/// @param input_file
/// @param output_file
/// @param check
/// @param displayCells
/// @param displayParticles
/// @return
template<int Dimension, typename ValueType, typename Array>
auto run_one_over_r(const int& tree_height, const int& group_size, const std::size_t order, Array const& pbc,
                    const int& nb_level_above_root, const std::string& input_file, const std::string& output_file,
                    const bool check, const bool displayCells, const bool displayParticles) -> void

{
    using value_type = ValueType;
    static constexpr int dimension = Dimension;

    // matrix kernel
    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

    // near field
    using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

    // far field
    using interpolation_type = interpolator_alias<value_type, dimension, far_matrix_kernel_type>;
    using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

    // fmm operators
    using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    // run main program
    run<dimension, value_type, fmm_operators_type>(tree_height, group_size, order, pbc, nb_level_above_root, input_file,
                                                   output_file, check, displayCells, displayParticles);
}

/// @brief
/// @tparam ValueType
/// @tparam Dimension
/// @param tree_height
/// @param group_size
/// @param order
/// @param pbc
/// @param nb_level_above_root
/// @param input_file
/// @param output_file
/// @param check
/// @param displayCells
/// @param displayParticles
/// @return
template<int Dimension, typename ValueType, typename Array>
auto run_grad_one_over_r(const int& tree_height, const int& group_size, const std::size_t order, Array const& pbc,
                         const int& nb_level_above_root, const std::string& input_file, const std::string& output_file,
                         const bool check, const bool displayCells, const bool displayParticles) -> void
{
    using value_type = ValueType;
    static constexpr int dimension = Dimension;

    // matrix kernel
    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
    using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;

    // near field
    using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

    // far field
    using interpolation_type = interpolator_alias<value_type, dimension, far_matrix_kernel_type>;
    using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

    // fmm operators
    using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    // run main program
    run<dimension, value_type, fmm_operators_type>(tree_height, group_size, order, pbc, nb_level_above_root, input_file,
                                                   output_file, check, displayCells, displayParticles);
}

/// @brief
/// @tparam ValueType
/// @tparam Dimension
/// @param tree_height
/// @param group_size
/// @param order
/// @param pbc
/// @param nb_level_above_root
/// @param input_file
/// @param output_file
/// @param check
/// @param displayCells
/// @param displayParticles
/// @return
template<int Dimension, typename ValueType, typename Array>
auto run_val_grad_one_over_r(const int& tree_height, const int& group_size, const std::size_t order, Array const& pbc,
                             const int& nb_level_above_root, const std::string& input_file,
                             const std::string& output_file, const bool check, const bool displayCells,
                             const bool displayParticles) -> void
{
    using value_type = ValueType;
    static constexpr int dimension = Dimension;

    // matrix kernel
    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;

    // near field
    using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

    // far field
    using interpolation_type = interpolator_alias<value_type, dimension, far_matrix_kernel_type>;
    using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;

    // fmm operators
    using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    // run main program
    run<dimension, value_type, fmm_operators_type>(tree_height, group_size, order, pbc, nb_level_above_root, input_file,
                                                   output_file, check, displayCells, displayParticles);
}

/// @brief
/// @tparam ValueType
/// @tparam Dimension
/// @param tree_height
/// @param group_size
/// @param order
/// @param pbc
/// @param nb_level_above_root
/// @param input_file
/// @param output_file
/// @param check
/// @param displayCells
/// @param displayParticles
/// @return
template<int Dimension, typename ValueType, typename Array>
auto run_dimension(const int& kernel_choice, const int& tree_height, const int& group_size, const std::size_t order,
                   Array const& pbc, const int& nb_level_above_root, const std::string& input_file,
                   const std::string& output_file, const bool check, const bool displayCells,
                   const bool displayParticles) -> void
{
    using value_type = ValueType;
    static constexpr int dimension = Dimension;

    switch(kernel_choice)
    {
    case 0:
    {
        run_one_over_r<dimension, value_type>(tree_height, group_size, order, pbc, nb_level_above_root, input_file,
                                              output_file, check, displayCells, displayParticles);
    }
    break;
    case 1:
    {
        run_grad_one_over_r<dimension, value_type>(tree_height, group_size, order, pbc, nb_level_above_root, input_file,
                                                   output_file, check, displayCells, displayParticles);
    }
    break;
    case 2:
    {
        run_val_grad_one_over_r<dimension, value_type>(tree_height, group_size, order, pbc, nb_level_above_root,
                                                       input_file, output_file, check, displayCells, displayParticles);
    }
    break;
    default:
    {
        run_one_over_r<dimension, value_type>(tree_height, group_size, order, pbc, nb_level_above_root, input_file,
                                              output_file, check, displayCells, displayParticles);
    }
    break;
    }
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::input_file(),
                                                    args::output_file(), args::tree_height{}, args::block_size{},
                                                    args::order{},   // args::thread_count{},
                                                    //, args::log_file{}, args::log_level{}
                                                    args::pbc{}, args::extended_tree_height{},   // periodicity
                                                    local_args::kernel{}, local_args::dimension{}, local_args::check{},
                                                    local_args::displayParticles{}, local_args::displayCells{}

    );

    parser.parse(argc, argv);
    const int dim{parser.get<local_args::dimension>()};
    std::cout << cpp_tools::colors::blue << "<params> Dimension : " << dim << cpp_tools::colors::reset << '\n';

    const int kernel_choice{parser.get<local_args::kernel>()};
    std::cout << cpp_tools::colors::blue << "<params> Kernel : " << kernel_choice << cpp_tools::colors::reset << '\n';

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
    std::vector<bool> pbc(dim, false);
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
    // const bool postreat(parser.exists<laplace::args::post_traitement>());
    // const bool dataSorted(parser.exists<isSorted>());
    //  const int matrix_type = parser.get<laplace::args::matrix_kernel>();
    // const int which_interp(parser.get<interpolator>());

    using value_type = double;

    switch(dim)
    {
    case 2:
    {
        static constexpr int dimension{2};

        run_dimension<dimension, value_type>(kernel_choice, tree_height, group_size, order, pbc, nb_level_above_root,
                                             input_file, output_file, check, displayCells, displayParticles);
    }
    break;
    case 3:
    {
        static constexpr int dimension{3};

        run_dimension<dimension, value_type>(kernel_choice, tree_height, group_size, order, pbc, nb_level_above_root,
                                             input_file, output_file, check, displayCells, displayParticles);
    }
    break;
    default:
    {
        std::cout << "Only the two-dimensional and three-dimensional cases are supported !" << std::endl;
    }
    break;
    }

    return 0;
}
