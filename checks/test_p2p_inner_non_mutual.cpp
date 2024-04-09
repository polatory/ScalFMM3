#include <array>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>

// scalfmm
#include "scalfmm/container/block.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/p2p.hpp"
#include "scalfmm/tree/group_of_views.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/accurater.hpp"

// cpp tools
#include "cpp_tools/cl_parser/tcli.hpp"
#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

namespace local_args
{
    struct nb_particles
    {
        cpp_tools::cl_parser::str_vec flags = {"--N", "--number-particles"};
        std::string description = "Numbre of particles to generate";
        using type = std::size_t;
        std::string input_hint = "int"; /*!< The input hint */
        type def = 1000;
    };

    struct data_type
    {
        cpp_tools::cl_parser::str_vec flags = {"--use-float"};
        using type = bool;
        std::string description = "To generate float distribution";
        enum
        {
            flagged
        };
    };

    struct dimension
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n\t1, 2, 3 or 4";
        using type = std::size_t;
        type def = 3;
    };

    struct kernel
    {
        cpp_tools::cl_parser::str_vec flags = {"--kernel", "--k"};
        std::string description = "Kernel : \n\t0 for 'one_over_r' 2, 1 for 'grad_one_over_r<dim>, 2 for "
                                  "'val_grad_one_over_r<dim>, (default) 'one_over_r'";
        using type = std::size_t;
        type def = 0;
    };

    struct nb_runs
    {
        cpp_tools::cl_parser::str_vec flags = {"--number-runs", "--nruns"};
        std::string description = "Number of runs to get average runtime values";
        using type = std::size_t;
        type def = 1;
    };
}   // namespace local_args

/**
 * @brief Prints the content of a leaf to the standard output.
 *
 * This function iterates over all particles within the specified leaf and prints their details to the standard output.
 *
 * @tparam LeafType The type of the leaf being printed.
 *
 * @param leaf The constant reference to the leaf whose content is to be printed.
 *
 */
template<typename LeafType>
auto print_leaf_content(LeafType const& leaf) -> void
{
    using leaf_type = LeafType;
    using const_proxy_type = typename leaf_type::const_proxy_type;

    // loop through all the particles of the leaf
    for(auto part: leaf)
    {
        auto p = const_proxy_type(part);
        std::cout << p << std::endl;
    }
}

/**
 * @brief Initializes the content of a leaf with random values.
 *
 * This function iterates over all particles in the specified leaf and initializes their positions, inputs, and outputs
 * with random and default values, respectively. Positions and inputs are assigned random values within a specified
 * range (0.0 to 2.0), while outputs are initialized to zero. Additionally, each particle is assigned a unique global
 * index starting from zero.
 *
 * @tparam LeafType The type of the leaf being initialized.
 *
 * @param leaf The leaf whose content is to be initialized. This operation modifies the leaf by setting
 * particle positions, inputs, and outputs to their initial values.
 */
template<typename LeafType>
auto init_leaf_content(LeafType& leaf) -> void
{
    using leaf_type = LeafType;
    using value_type = typename leaf_type::value_type;
    using particle_type = typename leaf_type::particle_type;
    using proxy_type = typename particle_type::proxy_type;

    using seed_generator_type = std::random_device;
    using random_generator_type = std::mt19937;
    using distribution_type = std::uniform_real_distribution<value_type>;

    static constexpr std::size_t dimension = particle_type::dimension;
    static constexpr std::size_t inputs_size = particle_type::inputs_size;
    static constexpr std::size_t outputs_size = particle_type::outputs_size;

    // set up the random generator
    seed_generator_type rd;
    random_generator_type gen(rd());
    distribution_type dis(0.0, 2.0);
    auto random_r = [&dis, &gen]() { return dis(gen); };

    // get the begin and end iterators of the leaf
    auto part_begin = leaf.begin();
    auto part_end = leaf.end();

    std::size_t idx{0};

    while(part_begin != part_end)
    {
        auto p = proxy_type(*part_begin);
        // initialize position
        for(std::size_t i = 0; i < dimension; ++i)
        {
            p.position(i) = random_r();
        }
        // initialize inputs
        for(std::size_t i = 0; i < inputs_size; ++i)
        {
            p.inputs(i) = random_r();
        }
        // initialize outputs
        for(std::size_t i = 0; i < outputs_size; ++i)
        {
            p.outputs(i) = value_type(0.);
        }
        // set the global index of the particle
        std::get<0>(p.variables()) = idx++;
        ++part_begin;
    }
}

/**
 * @brief Copies the content from one leaf to another.
 *
 * This function iterates over all particles in the source leaf (`src_leaf`) and copies their positions, inputs,
 * outputs, and the first variable to the corresponding particles in the destination leaf (`dst_leaf`). Both leaves
 * are expected to have the same type and contain the same number of particles.
 *
 * @tparam LeafType The type of the leaves involved in the copy operation.
 *
 * @param src_leaf The source leaf from which content is copied. This leaf remains unchanged.
 * @param dst_leaf The destination leaf to which content is copied. This leaf is modified to reflect the content
 * of `src_leaf`.
 */
template<typename LeafType>
auto copy_leaf_content(LeafType& src_leaf, LeafType& dst_leaf) -> void
{
    using leaf_type = LeafType;
    using value_type = typename leaf_type::value_type;
    using particle_type = typename leaf_type::particle_type;
    using proxy_type = typename particle_type::proxy_type;
    using const_proxy_type = typename particle_type::const_proxy_type;

    static constexpr std::size_t dimension = particle_type::dimension;
    static constexpr std::size_t inputs_size = particle_type::inputs_size;
    static constexpr std::size_t outputs_size = particle_type::outputs_size;

    // check at run-time that both leaves have the same number of particles
    assert(src_leaf.size() == dst_leaf.size());

    // get the begin and end iterators of the first leaf (= source leaf)
    auto src_part_begin = src_leaf.begin();
    auto src_part_end = src_leaf.end();

    // get the begin and end iterators of the second leaf (= destination leaf)
    auto dst_part_begin = dst_leaf.begin();
    auto dst_part_end = dst_leaf.end();

    while(src_part_begin != src_part_end)
    {
        // get a proxy on the current particle of the first leaf (= source leaf)
        auto src_p = proxy_type(*src_part_begin);
        // get a proxy on the current particle of the second leaf (= destination leaf)
        auto dst_p = proxy_type(*dst_part_begin);
        // copy source position to destination position
        for(std::size_t i = 0; i < dimension; ++i)
        {
            dst_p.position(i) = src_p.position(i);
        }
        // copy source inputs to destination inputs
        for(std::size_t i = 0; i < inputs_size; ++i)
        {
            dst_p.inputs(i) = src_p.inputs(i);
        }
        // copy source outputs to destination outputs
        for(std::size_t i = 0; i < outputs_size; ++i)
        {
            dst_p.outputs(i) = src_p.outputs(i);
        }
        // copy first source variable to first destination variable
        std::get<0>(dst_p.variables()) = std::get<0>(src_p.variables());

        ++src_part_begin;
        ++dst_part_begin;
    }
}

/**
 * @brief Compares the output values of two leaf objects and calculates the relative L2 norm error.
 *
 * This function iterates over two leaf objects of the same type, comparing their output values particle by particle.
 * It assumes both leaves contain the same number of particles and that each particle in the leaves has a set of
 * output values. The function calculates the error for each corresponding pair of output values between the two leaves
 * and accumulates these errors to compute the relative L2 norm error across all particles and their outputs.
 *
 * @tparam LeafType The type of the leaf objects being compared.
 *
 * @param lhs_leaf A constant reference to the first leaf object.
 * @param rhs_leaf A constant reference to the second leaf object.
 *
 */
template<typename LeafType>
auto compare_leaf_outputs(LeafType const& lhs_leaf, LeafType const& rhs_leaf) -> typename LeafType::value_type
{
    using leaf_type = LeafType;
    using value_type = typename leaf_type::value_type;
    using particle_type = typename leaf_type::particle_type;
    using proxy_type = typename particle_type::proxy_type;
    using const_proxy_type = typename particle_type::const_proxy_type;

    using accurater_type = scalfmm::utils::accurater<value_type>;

    static constexpr std::size_t outputs_size = particle_type::outputs_size;

    // check at run-time that both leaves have the same number of particles
    assert(lhs_leaf.size() == rhs_leaf.size());

    // get the begin and end iterators of the first leaf
    auto lhs_part_begin = lhs_leaf.begin();
    auto lhs_part_end = lhs_leaf.end();

    // get the begin and end iterators of the second leaf
    auto rhs_part_begin = rhs_leaf.begin();
    auto rhs_part_end = rhs_leaf.end();

    accurater_type error;

    while(lhs_part_begin != rhs_part_end)
    {
        // get a proxy on the current particle of the first leaf
        auto lhs_p = const_proxy_type(*lhs_part_begin);
        // get a proxy on the current particle of the second leaf
        auto rhs_p = const_proxy_type(*rhs_part_begin);

        // compute the error between the two outputs
        for(std::size_t i = 0; i < outputs_size; ++i)
        {
            error.add(lhs_p.outputs(i), rhs_p.outputs(i));
        }

        ++lhs_part_begin;
        ++rhs_part_begin;
    }

    // return only the relative l2 norm error
    return error.get_relative_l2_norm();
}

/**
 * @brief Creates a leaf view from a given group.
 *
 * This function initializes a leaf view object based on the provided group. It creates a new leaf of type `LeafType`
 * by extracting and utilizing the particle storage and symbolic information from the `GroupType` object. The function
 * directly modifies the `leaf` parameter to represent a view onto the first set of particles contained within the
 * `group`. It is designed to work with data structures where groups manage collections of particles and leaves act
 * as views or references to these collections for more efficient access and manipulation.
 *
 * @tparam GroupType The type of the group from which the leaf view is created.
 * @tparam LeafType The type of the leaf to be initialized.
 *
 * @param group A reference to the group object from which the leaf view is to be created. The group must contain
 * particle storage and symbolic information.
 * @param leaf A reference to the leaf object to be initialized. This function directly modifies `leaf` to represent
 * a view on a subset of particles from `group`.
 *
 */
template<typename GroupType, typename LeafType>
auto make_leaf_view_from_group(GroupType& group, LeafType& leaf) -> void
{
    using leaf_type = LeafType;

    // get storage and symbolic data of the provided group
    auto& particles_storage = group.storage();
    auto& group_symbolics = group.csymbolics();
    auto leaf_sym_ptr = &particles_storage.symbolics(0);
    std::size_t number_of_particles_in_group = group_symbolics.number_of_particles_in_group;

    // create a leaf with a view on the particle storage
    leaf =
      leaf_type(std::make_pair(particles_storage.begin(), particles_storage.begin() + number_of_particles_in_group),
                leaf_sym_ptr);
}

/**
 * @brief Executes the main program to benchmark and compare the performance of P2P operators.
 *
 * This function benchmarks three types of P2P operations: inner non-mutual, inner mutual, and outer. It performs
 * a specified number of runs for each operation, calculates the average execution time, and evaluates the L2 error
 * between the results of inner mutual and non-mutual operations. The function is templated to allow for flexibility
 * in the types of values and matrix kernels used in the computations, as well as the dimensionality of the problem.
 *
 * @tparam ValueType The data type for the values used in computations (e.g., float, double).
 * @tparam MatrixKernelType The type of the matrix kernel to be used in the P2P operations.
 * @tparam Dimension The spatial dimension of the particle system (e.g., 2 for 2D, 3 for 3D).
 *
 * @param N The number of particles in each group or leaf.
 * @param number_runs The number of times each P2P operation is executed for benchmarking.
 *
 * The function initializes four groups of particles and corresponding leaf views. It then fills these leaves with
 * particles and benchmarks the specified P2P operations, printing the average execution time and the L2 error between
 * the outputs of inner mutual and non-mutual operations. The function uses high-resolution timers to measure execution
 * time and outputs the results to the standard output.
 *
 */
template<typename ValueType, typename MatrixKernelType, std::size_t Dimension>
auto run(std::size_t N, std::size_t number_runs) -> void
{
    using value_type = ValueType;
    using matrix_kernel_type = MatrixKernelType;
    static constexpr std::size_t dimension = Dimension;

    // particle type
    static constexpr std::size_t nb_inputs = matrix_kernel_type::km;
    static constexpr std::size_t nb_outputs = matrix_kernel_type::kn;
    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs, std::size_t>;

    // leaf and group
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using group_type = scalfmm::component::group_of_particles<leaf_type, particle_type>;

    // time measurement
    using duration_type = std::chrono::nanoseconds;
    using timer_type = cpp_tools::timers::timer<duration_type>;

    // matrix kernel
    matrix_kernel_type mk{};
    std::cout << cpp_tools::colors::blue << "[param] kernel = " << mk.name() << cpp_tools::colors::reset << std::endl;

    // creation of groups with 1 component containing each N particles
    group_type group1(0, N, 1, N, 0, true);
    group_type group2(0, N, 1, N, 0, true);
    group_type group3(0, N, 1, N, 0, true);
    group_type group4(0, N, 1, N, 0, true);

    // creation of leaves from the groups (only views on the group's storage)
    leaf_type leaf1, leaf2, leaf3, leaf4;
    make_leaf_view_from_group(group1, leaf1);
    make_leaf_view_from_group(group2, leaf2);
    make_leaf_view_from_group(group3, leaf3);
    make_leaf_view_from_group(group4, leaf4);

    // fill the leaves with particles
    init_leaf_content(leaf1);
    copy_leaf_content(leaf1, leaf2);
    init_leaf_content(leaf3);
    init_leaf_content(leaf4);

    timer_type timer_p2p_inner_non_mutual{}, timer_p2p_inner_mutual{}, timer_p2p_outer{};
    value_type time_p2p_non_mutual{}, time_p2p_mutual{}, time_p2p_outer{}, l2_error{};
    std::size_t total_number_runs = (number_runs == 1) ? 1 : (number_runs - 1);

    // --- P2P inner non-mutual ---
    std::cout << cpp_tools::colors::green;
    std::cout << "\n --- BENCHMARK P2P inner non-mutual ---" << std::endl;
    for(std::size_t i = 0; i < number_runs; ++i)
    {
        // reset the output of the leaf
        group1.storage().reset_outputs();

        // run P2P operator
        timer_p2p_inner_non_mutual.tic();
        scalfmm::operators::p2p_inner_non_mutual(mk, leaf1);
        timer_p2p_inner_non_mutual.tac();

        // print elapsed time of the current iteration
        if(i == 0 && number_runs > 1)
        {
            std::cout << "\t- RUN " << i << " - elapsed time = " << timer_p2p_inner_non_mutual.elapsed() / 1e9
                      << " s (WARM-UP run - not taken into account for average time measurement)" << std::endl;
            timer_p2p_inner_non_mutual.reset();
        }
        else
        {
            std::cout << "\t- RUN " << i << " - elapsed time = " << timer_p2p_inner_non_mutual.elapsed() / 1e9 << " s"
                      << std::endl;
        }
    }
    std::cout << cpp_tools::colors::reset;

    // --- P2P inner mutual ---
    std::cout << cpp_tools::colors::green;
    std::cout << "\n --- BENCHMARK P2P inner mutual ---" << std::endl;
    for(std::size_t i = 0; i < number_runs; ++i)
    {
        // reset the output of the leaf
        group2.storage().reset_outputs();

        // run the P2P operator
        timer_p2p_inner_mutual.tic();
        scalfmm::operators::p2p_inner_mutual(mk, leaf2);
        timer_p2p_inner_mutual.tac();

        // print elapsed time of the current iteration
        if(i == 0 && number_runs > 1)
        {
            std::cout << "\t- RUN " << i << " - elapsed time = " << timer_p2p_inner_mutual.elapsed() / 1e9
                      << " s (WARM-UP run - not taken into account for average time measurement)" << std::endl;
            timer_p2p_inner_mutual.reset();
        }
        else
        {
            std::cout << "\t- RUN " << i << " - elapsed time = " << timer_p2p_inner_mutual.elapsed() / 1e9 << " s"
                      << std::endl;
        }
    }
    std::cout << cpp_tools::colors::reset;

    // --- P2P outer ---
    std::array<std::size_t, 1> dummy{};
    std::cout << cpp_tools::colors::green;
    std::cout << "\n --- BENCHMARK P2P outer ---" << std::endl;
    for(std::size_t i = 0; i < number_runs; ++i)
    {
        // reset the output of the leaves
        group3.storage().reset_outputs();
        group4.storage().reset_outputs();

        // run the P2P operator
        timer_p2p_outer.tic();
        scalfmm::operators::p2p_outer(mk, leaf3, leaf4, dummy);
        timer_p2p_outer.tac();

        // print elapsed time of the current iteration
        if(i == 0 && number_runs > 1)
        {
            std::cout << "\t- RUN " << i << " - elapsed time = " << timer_p2p_outer.elapsed() / 1e9
                      << " s (WARM-UP run - not taken into account for average time measurement)" << std::endl;
            timer_p2p_outer.reset();
        }
        else
        {
            std::cout << "\t- RUN " << i << " - elapsed time = " << timer_p2p_outer.elapsed() / 1e9 << " s"
                      << std::endl;
        }
    }
    std::cout << cpp_tools::colors::reset;

    // post-processing
    time_p2p_non_mutual = timer_p2p_inner_non_mutual.cumulated() / 1e9 / total_number_runs;
    time_p2p_mutual = timer_p2p_inner_mutual.cumulated() / 1e9 / total_number_runs;
    time_p2p_outer = timer_p2p_outer.cumulated() / 1e9 / total_number_runs;
    l2_error = compare_leaf_outputs(leaf1, leaf2);

    // print results and performances
    std::cout << "\n[avg time][p2p inner non-mutual] = " << time_p2p_non_mutual << std::endl;
    std::cout << "[avg time][p2p inner mutual] = " << time_p2p_mutual << std::endl;
    std::cout << "[avg time][p2p outer] = " << time_p2p_outer << std::endl;
    std::cout << "[ratio][p2p inner non-mutual | p2p outer] = " << time_p2p_non_mutual / time_p2p_outer << std::endl;
    std::cout << std::endl;
    std::cout << "[error][p2p inner mutual | p2p inner non-mutual] = " << l2_error << std::endl;
}

/**
 * @brief Executes the main program for a specified dimension and kernel choice.
 *
 * This function selects and runs a specific kernel based on the `kernel_choice` parameter for simulations in a given
 * dimension `Dimension`.
 *
 * @tparam Dimension The spatial dimension of the simulation.
 * @tparam ValueType The data type used for computation, typically a floating-point type like `float` or `double`.
 *
 * @param kernel_choice An index representing the choice of kernel to run. Each index maps to a specific kernel
 * implementation within the Laplace domain:
 * - 0: `one_over_r`
 * - 1: `grad_one_over_r`
 * - 2: `val_grad_one_over_r`
 * Any other value will result in an error indicating an unsupported kernel option.
 *
 * @param N The number of elements or particles involved in the simulation.
 * @param nb_runs The number of times the simulation is executed. Multiple runs can be used for benchmarking or
 * averaging performance metrics.
 *
 */
template<std::size_t Dimension, typename ValueType>
auto run_dimension(std::size_t kernel_choice, std::size_t N, std::size_t nb_runs) -> void
{
    using value_type = ValueType;
    static constexpr std::size_t dimension = Dimension;

    switch(kernel_choice)
    {
    case 0:
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        run<value_type, matrix_kernel_type, dimension>(N, nb_runs);
    }
    break;
    case 1:
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
        run<value_type, matrix_kernel_type, dimension>(N, nb_runs);
    }
    break;
    case 2:
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
        run<value_type, matrix_kernel_type, dimension>(N, nb_runs);
    }
    break;
    default:
    {
        std::cerr << "Unsupported kernel option = " << kernel_choice << "!" << std::endl;
    }
    break;
    }
}

/**
 * @brief Executes a simulation run with a specified value type, dimension, kernel choice, number of particles, and
 number of runs.

 * @tparam ValueType The data type used for computation, typically a floating-point type like `float` or `double`.
 *
 * @param dim The dimensionality of the simulation (1D, 2D, 3D, or 4D).
 * @param kernel_choice An index representing the choice of kernel to run. Each index maps to a specific kernel
 * implementation within the Laplace domain:
 * - 0: `one_over_r`
 * - 1: `grad_one_over_r`
 * - 2: `val_grad_one_over_r`
 * Any other value will result in an error indicating an unsupported kernel option.
 *
 * @param N The number of elements or particles involved in the simulation.
 * @param nb_runs The number of times the simulation is executed. Multiple runs can be used for benchmarking or
 * averaging performance metrics.
 */
template<typename ValueType>
auto run_value_type(std::size_t dim, std::size_t kernel_choice, std::size_t N, std::size_t nb_runs) -> void
{
    using value_type = ValueType;

    std::cout << cpp_tools::colors::blue << "[param] dimension = " << dim << cpp_tools::colors::reset << std::endl;

    switch(dim)
    {
    case 1:
    {
        static constexpr std::size_t dimension{1};
        run_dimension<dimension, value_type>(kernel_choice, N, nb_runs);
    }
    break;
    case 2:
    {
        static constexpr std::size_t dimension{2};
        run_dimension<dimension, value_type>(kernel_choice, N, nb_runs);
    }
    break;
    case 3:
    {
        static constexpr std::size_t dimension{3};
        run_dimension<dimension, value_type>(kernel_choice, N, nb_runs);
    }
    break;
    case 4:
    {
        static constexpr std::size_t dimension{4};
        run_dimension<dimension, value_type>(kernel_choice, N, nb_runs);
    }
    break;
    default:
    {
        std::cerr << "Only the 1D, 2D, 3D and 4D cases are supported !" << std::endl;
    }
    break;
    }
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, local_args::nb_particles{}, local_args::nb_runs{},
                                        local_args::dimension{}, local_args::kernel{}, local_args::data_type{});
    parser.parse(argc, argv);

    // get the parameter values
    const std::size_t N{parser.get<local_args::nb_particles>()};
    const std::size_t dim{parser.get<local_args::dimension>()};
    const std::size_t nb_runs{parser.get<local_args::nb_runs>()};
    const std::size_t kernel_choice{parser.get<local_args::kernel>()};
    const bool use_double = (parser.exists<local_args::data_type>() ? false : true);

    std::cout << cpp_tools::colors::blue;
    std::cout << "[param] N = " << N << std::endl;
    std::cout << "[param] nb-runs = " << nb_runs << std::endl;
    std::cout << cpp_tools::colors::reset;

    if(use_double)
    {
        using value_type = double;
        std::cout << cpp_tools::colors::blue << "[param] data-type = double" << cpp_tools::colors::reset << std::endl;
        run_value_type<value_type>(dim, kernel_choice, N, nb_runs);
    }
    else
    {
        using value_type = float;
        std::cout << cpp_tools::colors::blue << "[param] data-type = float" << cpp_tools::colors::reset << std::endl;
        run_value_type<value_type>(dim, kernel_choice, N, nb_runs);
    }

    return 0;
}
