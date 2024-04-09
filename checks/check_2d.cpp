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
// ./check/Release/check_2d --input-file check_per.fma  -o 4 --tree-height 4 -gs 2
// @FUSE_FFTW
// @FUSE_CBLAS
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
// Scalfmm includes
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/tools/fma_loader.hpp"
// Tree
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"

// replicated container
#include "scalfmm/meta/utils.hpp"
// parameters
#include "scalfmm/utils/parameters.hpp"
// Fmm operators
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/io_helpers.hpp"

//
#include "scalfmm/tools/tree_io.hpp"
#include "scalfmm/utils/compare_trees.hpp"

/// \code {.c++}
/// check/Release/check_2d - th 4 --order 5 --input - file../data/units/test_2d_ref.fma -gs 100 --check
/// \endcode
///

namespace local_args
{
    struct matrix_kernel
    {
        cpp_tools::cl_parser::str_vec flags = {"--kernel", "-k"};
        const char *description = "Matrix kernels: \n   0) 1/r, 1) grad(ln r), "
                                  "2)  shift(ln r)-> grad  3) val_grad( 1/r)";
        using type = int;
        type def = 0;
    };

    struct check
    {
        cpp_tools::cl_parser::str_vec flags = {"--check"};
        const char *description = "Check with p2p ";
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
        const char *description = "Display all particles  ";
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
        const char *description = "Display the cells at leaf level  ";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
} // namespace local_args

template <typename Container> auto read_data(const std::string &filename)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using value_type = typename particle_type::position_value_type;
    static constexpr std::size_t dimension{particle_type::dimension};
    const bool verbose{true};

    scalfmm::io::FFmaGenericLoader<value_type, dimension> loader(filename, verbose);

    const std::size_t number_of_particles{loader.getNumberOfParticles()};
    std::cout << cpp_tools::colors::yellow
              << "[file][n_particles] : " << number_of_particles
              << cpp_tools::colors::reset << '\n';
    const auto width{loader.getBoxWidth()};
    std::cout << cpp_tools::colors::yellow << "[file][box_width] : " << width
              << cpp_tools::colors::reset << '\n';
    const auto center{loader.getBoxCenter()};
    std::cout << cpp_tools::colors::yellow << "[file][box_centre] : " << center
              << cpp_tools::colors::reset << '\n';

    auto nb_val_to_red_per_part = loader.getNbRecordPerline();
    // could be a problem for binary file (float double)
    std::vector<value_type> values_to_read(nb_val_to_red_per_part);

    container_type container(number_of_particles);

    for (std::size_t idx = 0; idx < number_of_particles; ++idx)
        {
            loader.fillParticle(values_to_read.data(), nb_val_to_red_per_part);
            particle_type p;
            std::size_t ii{0};
            for (auto &e : p.position())
                {
                    e = values_to_read[ii++];
                }
            for (auto &e : p.inputs())
                {
                    e = values_to_read[ii++];
                }
            // p.variables(values_to_read[ii++], idx, 1);
            p.variables(idx);
            container.insert_particle(idx, p);
        }
    return std::make_tuple(container, center, width);
}

/**
 * @brief
 *
 * @param tree
 * @param ref
 */

template <typename Tree, typename Container>
auto check_output(Container const &part, Tree const &tree)
    -> scalfmm::utils::accurater<double>
{
    scalfmm::utils::accurater<double> error;

    auto nb_out = Container::value_type::outputs_size;
    std::cout << "number of output " << nb_out << std::endl;
    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&nb_out, &part, &error](auto& leaf)
                                      {
                                          //   const auto nb_elt = leaf.size();
                                          for(auto const p_tuple_ref: leaf)
                                          {
                                              // We construct a particle type for classical acces
                                              const auto& p = typename Tree::leaf_type::const_proxy_type(p_tuple_ref);

                                              const auto& idx = std::get<0>(p.variables());
                                              //   std::cout << "fmm    " << std::setprecision(8) << p << "\n";
                                              //   std::cout << "direct " << std::setprecision(8) << part.particle(idx)
                                              //             << "\n";

                                              auto output = p.outputs();
                                              auto output_ref = part.particle(idx).outputs();

                                              //   auto diff{output_ref};

                                              for(std::size_t i{0}; i < nb_out; ++i)
                                              {
                                                  error.add(output_ref.at(i), output.at(i));
                                              }
                                              //   std::cout << "Error: ";
                                              //   scalfmm::io::print(std::cout, diff);
                                              //   std::cout << std::endl;
                                          }
                                      });
    return error;
}
template <int Dimension, typename value_type, class FMM_OPERATOR_TYPE>
auto run(const int &tree_height, const int &group_size, const std::size_t order,
         const std::string &input_file, const std::string &output_file,
         const bool check, const bool displayCells, const bool displayParticles)
    -> int

{
    using near_matrix_kernel_type =
        typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;

    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};

    // position
    // number of input values and type
    // number of output values
    // variables original index
    using particle_type =
        scalfmm::container::particle<value_type, Dimension, value_type,
                                     nb_inputs_near, value_type,
                                     nb_outputs_near, std::size_t>;
    using container_type =
        scalfmm::container::particle_container<particle_type>;

    using position_type = typename particle_type::position_type;
    //

    using near_matrix_kernel_type =
        typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;
    using far_field_type = typename FMM_OPERATOR_TYPE::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;

    using far_matrix_kernel_type =
        typename interpolator_type::matrix_kernel_type;
    //
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;
    ///////////////////////////////////////////////////////////////
    // Particles read
    position_type box_center{};
    value_type box_width{};
    container_type container{};
    std::tie(container, box_center, box_width) =
        read_data<container_type>(input_file);

    //  read_data<Dimension>(input_file, container, box_center, box_width);
    //
    ////////////////////////////////////////////////////
    // Build tree
    box_type box(box_width, box_center);
    bool sorted = false;
    group_tree_type tree(tree_height, order, box, group_size, group_size,
                         container, sorted);
    //
    ////////////////////////////////////////////////////

    ///////////////////////////////////////////////////
    // Periodic run
    far_matrix_kernel_type mk_far{};
    auto total_height = tree_height;
    interpolator_type interpolator(mk_far, order, total_height, box_width);

    typename FMM_OPERATOR_TYPE::far_field_type far_field(interpolator);
    //
    near_matrix_kernel_type mk_near{};
    const bool mutual_near = true;

    typename FMM_OPERATOR_TYPE::near_field_type near_field(mk_near,mutual_near);
    //
    std::cout << cpp_tools::colors::blue << "Fmm with kernels: " << std::endl
              << "       near " << mk_near.name() << std::endl
              << "       far  " << mk_far.name() << std::endl
              << cpp_tools::colors::reset;

    FMM_OPERATOR_TYPE fmm_operator(near_field, far_field);
    //
    // Build interaction lists
     int const & separation_criterion = fmm_operator.near_field().separation_criterion();
     bool const &  mutual = fmm_operator.near_field().mutual();
    scalfmm::list::sequential::build_interaction_lists(tree, tree, separation_criterion, mutual);

    auto operator_to_proceed = scalfmm::algorithms::all;
    std::cout << cpp_tools::colors::blue << "operator_to_proceed: ";
    scalfmm::algorithms::print(operator_to_proceed);
    std::cout << cpp_tools::colors::reset << std::endl;

    scalfmm::algorithms::sequential::sequential(tree, fmm_operator, operator_to_proceed);
    if(check)
    {
            scalfmm::algorithms::full_direct(container, mk_near);

            auto error1 = check_output(container, tree);
            std::cout << error1 << std::endl;
    }
        if(displayParticles)
        {
            scalfmm::component::for_each_leaf(std::begin(tree), std::end(tree),
                                              [&box, tree_height](auto& leaf)
                                              {
                                                  scalfmm::io::print_leaf(leaf, box, tree_height - 1);
                                                  std::cout << "-----\n\n";
                                              });
        }
    
    if(!output_file.empty())
    {
        std::cout << "Write outputs in " << output_file << std::endl;
        scalfmm::io::FFmaGenericWriter<value_type> writer(output_file);
        writer.writeDataFromTree(tree, container.size());
    }
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Save Tree\n";
    std::string outName("saveTree.bin");
    std::string header("Uniform FFT ");
    scalfmm::tools::io::save(outName, tree, header);
    //
    std::cout << "Read Tree\n";
    value_type eps{1.e-8};
    auto tree1 = scalfmm::tools::io::read<group_tree_type>(outName);

    if(scalfmm::utils::compare_two_trees(tree, tree1, eps, 3))
    {
        std::cout << "Same trees !\n";
    }
    else
    {
        std::cout << "Trees are different!\n";
    }
    return 1;
}

template <int Dimension, typename value_type>
auto run_general(const int &tree_height, const int &group_size,
                 const std::size_t order, const int kernel,
                 const std::string &input_file, const std::string &output_file,
                 const bool check, const bool displayCells,
                 const bool displayParticles) -> int

{
    //
    // using far_matrix_kernel_type =
    // scalfmm::matrix_kernels::others::one_over_r2; using
    // near_matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
    //  using options = scalfmm::options::uniform_<scalfmm::options::fft_>;
    using options = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;
    // using options = scalfmm::options::chebyshev_<scalfmm::options::dense_>;
    if(kernel == 0)
    {   // one_over_r
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, Dimension, far_matrix_kernel_type, options>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        return run<Dimension, value_type, fmm_operators_type>(tree_height, group_size, order, input_file, output_file,
                                                              check, displayCells, displayParticles);
    }
    else if (kernel == 1)
        { // grad_ln_r
            using far_matrix_kernel_type =
                scalfmm::matrix_kernels::laplace::grad_ln_2d;
            using near_matrix_kernel_type =
                scalfmm::matrix_kernels::laplace::grad_ln_2d;
            using near_field_type = scalfmm::operators::near_field_operator<
                near_matrix_kernel_type>;
            //
            using interpolation_type =
                scalfmm::interpolation::interpolator<
                    value_type, Dimension, far_matrix_kernel_type, options>;
            using far_field_type =
                scalfmm::operators::far_field_operator<interpolation_type,
                                                       false>;

            using fmm_operators_type =
                scalfmm::operators::fmm_operators<near_field_type,
                                                  far_field_type>;

            return run<Dimension, value_type, fmm_operators_type>(
                tree_height, group_size, order, input_file, output_file, check,
                displayCells, displayParticles);
        }
    else if (kernel == 2)
        { // shift_ln_r
            using far_matrix_kernel_type =
                scalfmm::matrix_kernels::laplace::ln_2d;
            using near_matrix_kernel_type =
                scalfmm::matrix_kernels::laplace::grad_ln_2d;
            using near_field_type = scalfmm::operators::near_field_operator<
                near_matrix_kernel_type>;
            //
            using interpolation_type =
                scalfmm::interpolation::interpolator<
                    value_type, Dimension, far_matrix_kernel_type, options>;
            using far_field_type =
                scalfmm::operators::far_field_operator<interpolation_type,
                                                       true>;

            using fmm_operators_type =
                scalfmm::operators::fmm_operators<near_field_type,
                                                  far_field_type>;

            return run<Dimension, value_type, fmm_operators_type>(
                tree_height, group_size, order, input_file, output_file, check,
                displayCells, displayParticles);
        }
    else if (kernel == 3)
        { // val_grad_one_over_r
            using far_matrix_kernel_type =
                scalfmm::matrix_kernels::laplace::val_grad_one_over_r<2>;
            using near_matrix_kernel_type =
                scalfmm::matrix_kernels::laplace::val_grad_one_over_r<2>;
            using near_field_type = scalfmm::operators::near_field_operator<
                near_matrix_kernel_type>;
            //
            using interpolation_type =
                scalfmm::interpolation::interpolator<
                    value_type, Dimension, far_matrix_kernel_type, options>;
            using far_field_type =
                scalfmm::operators::far_field_operator<interpolation_type,
                                                       false>;

            using fmm_operators_type =
                scalfmm::operators::fmm_operators<near_field_type,
                                                  far_field_type>;

            return run<Dimension, value_type, fmm_operators_type>(
                tree_height, group_size, order, input_file, output_file, check,
                displayCells, displayParticles);
        }
    else
    {
        return 0;
    }
    //   scalfmm::matrix_kernels::laplace::one_over_r;
    //   scalfmm::matrix_kernels::laplace::grad_ln_2d;
}
auto main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) -> int
{ //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
        cpp_tools::cl_parser::help{}, args::input_file(), args::output_file(),
        args::tree_height{}, args::block_size{},
        args::order{}, // args::thread_count{},
        //, args::log_file{}, args::log_level{}
        local_args::matrix_kernel{}, // periodicity
        /*local_args::dimension{}, */ local_args::check{},
        local_args::displayParticles{}, local_args::displayCells{}

    );

    parser.parse(argc, argv);
    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue
              << "<params> Tree height : " << tree_height
              << cpp_tools::colors::reset << '\n';

    const int group_size{parser.get<args::block_size>()};
    std::cout << cpp_tools::colors::blue
              << "<params> Group Size : " << group_size
              << cpp_tools::colors::reset << '\n';

    const std::string input_file{parser.get<args::input_file>()};
    if (!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue
                      << "<params> Input file : " << input_file
                      << cpp_tools::colors::reset << '\n';
        }

    const std::string output_file{parser.get<args::output_file>()};
    if (!output_file.empty())
        {
            std::cout << cpp_tools::colors::blue
                      << "<params> Output file : " << output_file
                      << cpp_tools::colors::reset << '\n';
        }
    const auto order{parser.get<args::order>()};

    bool check{parser.exists<local_args::check>()};
    bool displayCells{parser.exists<local_args::displayCells>()};
    bool displayParticles{parser.exists<local_args::displayParticles>()};
    const int dimension{2 /*parser.get<local_args::dimension>()*/};
    const int kernel{parser.get<local_args::matrix_kernel>()};
    //
    using value_type = double;

    run_general<dimension, value_type>(tree_height, group_size, order, kernel, input_file, output_file, check,
                                       displayCells, displayParticles);
    return 0;
}
