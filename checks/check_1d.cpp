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
// ./checks/Release/check_1d --input-file ../data/units/test_1d_ref.fma  -o 4 --tree-height 4 -gs 2
//
// @FUSE_FFTW
// @FUSE_CBLAS

#include <algorithm>
#include <array>
#include <random>
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
#include "scalfmm/tree/tree.hpp"
//
#include "scalfmm/lists/lists.hpp"
// replicated container
#include "scalfmm/meta/utils.hpp"
// Fmm operators
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/utils/accurater.hpp"

//
namespace local_args
{
    struct matrix_kernel
    {
        cpp_tools::cl_parser::str_vec flags = {"--kernel", "-k"};
        std::string description = "Matrix kernels: \n   0) 1/x, 2) 1/r^2, "
                                  "2)  val_grad( 1/r)";
        using type = int;
        type def = 0;
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
    /**
     * @brief
     *
     */
    struct input_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
        type def = "";
    };
    /**
     * @brief
     *
     */
    struct nbPart
    {
        cpp_tools::cl_parser::str_vec flags = {"--num-part", "-n"};
        std::string description = "Number of particles).";
        using type = int;
        type def = 10000;
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
        particle_type p{};

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
        p.variables(idx);
        container.insert_particle(idx, p);
    }
    return std::make_tuple(container, center, width);
}

/// @brief  generate N  random particles in [-1,1]

/// @tparam Container
/// @param size
/// @return
template<typename Container>
auto generate_particles(const int N)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using point_type = typename particle_type::position_type;
    using value_type = typename particle_type::position_value_type;

    static constexpr std::size_t dimension{particle_type::dimension};

    container_type container(N);
    point_type box_center{static_cast<value_type>(0.0)};
    value_type box_width{static_cast<value_type>(2.0)};

    std::cout << cpp_tools::colors::yellow << '\n';
    std::cout << "[generate][nb-particles] : " << N << '\n';
    std::cout << "[generate][box-width]    : " << box_width << '\n';
    std::cout << "[generate][box-center]   : " << box_center << '\n';
    std::cout << cpp_tools::colors::reset << '\n';

    // random generator
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(123);
    std::uniform_real_distribution<value_type> dis(-1., 1.);
    auto random_r = [&dis, &gen]() { return dis(gen); };

    // inserting particles in the container
    for(std::size_t idx = 0; idx < N; ++idx)
    {
        // particle_type p;
        particle_type p{};
        for(std::size_t d = 0; d < dimension; ++d)
        {
            p.position(d) = box_center[d] + random_r() * 0.5 * box_width;
        }
        for(auto& e: p.inputs())
        {
            e = random_r();
        }
        for(auto& e: p.outputs())
        {
            e = value_type(0.);
        }
        p.variables(idx);
        container.insert_particle(idx, p);
    }

    return std::make_tuple(container, box_center, box_width);
}
/**
 * @brief
 *
 * @param tree
 * @param ref
 */
template<typename Tree, typename Container>
auto check_output(Container const& part, Tree const& tree) -> scalfmm::utils::accurater<double>
{
    scalfmm::utils::accurater<double> error;

    const int nb_out = Container::value_type::outputs_size;
    std::cout << "number of output " << nb_out << std::endl;
    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&nb_out, &part, &error](auto& leaf)
                                      {
                                          int i{0};
                                          for(auto const p_tuple_ref: leaf)
                                          {
                                              // We construct a particle type for classical acces
                                              const auto& p = typename Tree::leaf_type::const_proxy_type(p_tuple_ref);

                                              const auto& idx = std::get<0>(p.variables());

                                              auto output = p.outputs();
                                              auto output_ref = part.particle(idx).outputs();
                                              std::cout << i << " p_tree " << p.position() << " p_ref "
                                                        << part.particle(idx).position() << "  ";

                                              for(int n{0}; n < nb_out; ++n)
                                              {
                                                  std::cout << " (" << output_ref.at(n) << "  " << output.at(n)
                                                            << " ) ";
                                                  error.add(output_ref.at(n), output.at(n));
                                              }
                                              std::cout << std::endl;
                                          }
                                      });
    return error;
}
template<int Dimension, typename value_type, class FMM_OPERATOR_TYPE>
auto run(const int& tree_height, const int& group_size, const std::size_t order, const int N,
         const std::string& input_file, const std::string& output_file, const bool check, const bool displayCells,
         const bool displayParticles) -> int

{
    using near_matrix_kernel_type = typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;

    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};

    // position
    // number of input values and type
    // number of output values
    // variables original index
    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, std::size_t>;
    // using container_type = std::vector<particle_type>;
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
    if(!input_file.empty())
    {
        std::tie(container, box_center, box_width) = read_data<container_type>(input_file);
    }
    else
    {
        std::tie(container, box_center, box_width) = generate_particles<container_type>(N);
    }
    //
    ////////////////////////////////////////////////////
    // Build tree
    box_type box(box_width, box_center);
    bool sorted = false;
    group_tree_type tree(tree_height, order, box, group_size, group_size, container, sorted);
    //
    std::cout << "numbers of\n"
              << "    particles " << tree.number_of_particles() << std::endl
              << "    leaves    " << tree.number_of_leaves() << std::endl
              << "    cells     " << tree.number_of_cells() << std::endl;

    //
    ////////////////////////////////////////////////////

    ///////////////////////////////////////////////////
    far_matrix_kernel_type mk_far{};
    auto total_height = tree_height;
    interpolator_type interpolator(mk_far, order, total_height, box_width);

    // auto memory = interpolator.memory_usage();
    // std::cout << "memory " << memory << std::endl;

    typename FMM_OPERATOR_TYPE::far_field_type far_field(interpolator);
    //
    near_matrix_kernel_type mk_near{};
    typename FMM_OPERATOR_TYPE::near_field_type near_field(mk_near);
    //
    std::cout << cpp_tools::colors::blue << "Fmm with kernels: " << std::endl
              << "       near " << mk_near.name() << std::endl
              << "       far  " << mk_far.name() << std::endl
              << cpp_tools::colors::reset;

    FMM_OPERATOR_TYPE fmm_operator(near_field, far_field);
    //
    const int separation_criterion = fmm_operator.near_field().separation_criterion();
    const bool mutual = true;
    scalfmm::list::sequential::build_interaction_lists(tree, tree, fmm_operator);
    auto operator_to_proceed = scalfmm::algorithms::all;

    scalfmm::algorithms::sequential::sequential(tree, fmm_operator, operator_to_proceed);
    if(check)
    {
        // scalfmm::algorithms::full_direct(std::begin(container), std::end(container), mk_near);
        scalfmm::algorithms::full_direct(container, mk_near);

        auto error1 = check_output(container, tree);
        std::cout << error1 << std::endl;
    }
    if(displayCells)
    {
        scalfmm::component::for_each_leaf(std::begin(tree), std::end(tree),
                                          [&box, tree_height](auto& leaf)
                                          {
                                              //   const auto nb_elt = leaf.size();
                                              scalfmm::io::print_leaf(leaf, box, tree_height - 1);
                                              std::cout << "-----\n\n";
                                          });
    }
    return 1;
}

using option = scalfmm::options::uniform_<scalfmm::options::fft_>;

template<typename V, std::size_t D, typename MK>
using interpolator_alias = scalfmm::interpolation::interpolator<V, D, MK, option>;

template<int Dimension, typename value_type>
auto run_general(const int& tree_height, const int& group_size, const std::size_t order, const int kernel, const int N,
                 const std::string& input_file, const std::string& output_file, const bool check,
                 const bool displayCells, const bool displayParticles)
{
    //
    // using far_matrix_kernel_type =
    // scalfmm::matrix_kernels::others::one_over_r2; using
    // near_matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
    if(kernel == 0)
    {   // one_over_x
        using far_matrix_kernel_type = scalfmm::matrix_kernels::one_d ::one_over_x;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::one_d ::one_over_x;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolation_type = interpolator_alias<double, Dimension, far_matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        return run<Dimension, value_type, fmm_operators_type>(tree_height, group_size, order, N, input_file,
                                                              output_file, check, displayCells, displayParticles);
    }
    else if(kernel == -1)
    {   // one_over_x
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace ::one_over_r;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace ::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolation_type = interpolator_alias<double, Dimension, far_matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        return run<Dimension, value_type, fmm_operators_type>(tree_height, group_size, order, N, input_file,
                                                              output_file, check, displayCells, displayParticles);
    }
    else if(kernel == 1)
    {   // 1/r^2
        using far_matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolation_type = interpolator_alias<double, Dimension, far_matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        return run<Dimension, value_type, fmm_operators_type>(tree_height, group_size, order, N, input_file,
                                                              output_file, check, displayCells, displayParticles);
    }
    else if(kernel == 2)
    {   // val_grad_one_over_r
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<1>;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<1>;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using interpolation_type = interpolator_alias<double, Dimension, far_matrix_kernel_type>;

        // using interpolation_type =
        //   scalfmm::interpolation::uniform_interpolator<value_type, Dimension, far_matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        return run<Dimension, value_type, fmm_operators_type>(tree_height, group_size, order, N, input_file,
                                                              output_file, check, displayCells, displayParticles);
    }
    else
    {
        std::cout << " kernel < 4" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

//   scalfmm::matrix_kernels::laplace::one_over_r;
//   scalfmm::matrix_kernels::laplace::grad_ln_2d;

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{   //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, local_args::nbPart{}, local_args::input_file(), args::output_file(),
      args::tree_height{}, args::block_size{}, args::order{},   // args::thread_count{},
      //, args::log_file{}, args::log_level{}
      local_args::matrix_kernel{},   // periodicity
      /*local_args::dimension{}, */ local_args::check{}, local_args::displayParticles{}, local_args::displayCells{});

    parser.parse(argc, argv);
    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';

    const int group_size{parser.get<args::block_size>()};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';

    const std::string input_file{parser.get<local_args::input_file>()};
    int N{0};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                  << '\n';
    }
    else
    {
        N = parser.get<local_args::nbPart>();
    }

    const std::string output_file{parser.get<args::output_file>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    const auto order{parser.get<args::order>()};

    bool check{parser.exists<local_args::check>()};
    bool displayCells{parser.exists<local_args::displayCells>()};
    bool displayParticles{parser.exists<local_args::displayParticles>()};
    constexpr int dimension{1 /*parser.get<local_args::dimension>()*/};
    const int kernel{parser.get<local_args::matrix_kernel>()};
    //
    using value_type = double;

    auto ret = run_general<dimension, value_type>(tree_height, group_size, order, kernel, N, input_file, output_file,
                                                  check, displayCells, displayParticles);
    return ret;
}