// --------------------------------
// File: bench/fmm-computation.hpp
// --------------------------------

// scalfmm
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/fmm_operators.hpp"

#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/tree_io.hpp"

#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/io.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"

#include "scalfmm/utils/accurater.hpp"

#include "scalfmm/matrix_kernels/debug.hpp"
#include "scalfmm/matrix_kernels/gaussian.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"

#include "scalfmm/utils/parameters.hpp"

// cpp tools
#include <cpp_tools/cl_parser/cl_parser.hpp>

// STL
#include <random>
#include <vector>

namespace local_utils
{
    // Primary template handles matrix kernels that are not gaussian kernels
    template<typename MatrixKernelType>
    struct is_gaussian_kernel : std::false_type
    {
    };

    // Partial specialization recognizes gaussian kernels
    template<typename MatrixKernelType>
    struct is_gaussian_kernel<scalfmm::matrix_kernels::gaussian<MatrixKernelType>> : std::true_type
    {
    };

    // Helper alias
    template<typename MatrixKernelType>
    inline static constexpr bool is_gaussian_kernel_v = is_gaussian_kernel<MatrixKernelType>::value;
}   // namespace local_utils

namespace local_args
{
    struct interp_settings : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--interp-settings"};
        std::string description = "Interpolation settings:\n0) uniform (dense)\n1) uniform (low-rank)\n2) uniform "
                                  "(fft)\n3) chebyshev (dense)\n4) chebyshev (low-rank)\n";
        std::string input_hint = "int";
        using type = int;
        type def = 0;
    };

    struct kernel : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--kernel", "-k"};
        std::string description =
          "Matrix kernel:"
          "\n0) one_over_r\n1) one_over_r [non-symmetric]\n2) one_over_r [non-homogenous]"
          "\n3) one_over_r [non-homogenous - non-symmetric]\n4) grad_one_over_r<d>\n5) val_grad_one_over_r<d>"
          "\n6) grad_one_over_r<d> (optimized)\n7) val_grad_one_over_r<d> (optimized)"
          "\n8) gaussian (coeff=2.0) \n";
        std::string input_hint = "int";
        using type = int;
        type def = 0;
    };

    struct input_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
        type def = "";
    };

    struct output_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
        type def = "";
    };

    struct direct_computation
    {
        cpp_tools::cl_parser::str_vec flags = {"--direct-computation", "-direct"};
        std::string description = "Enable direct computation for the experiment.";
        using type = bool;
        enum
        {
            flagged
        };
    };

    struct fmm_computation
    {
        cpp_tools::cl_parser::str_vec flags = {"--fmm-computation", "-fmm"};
        std::string description = "Enable fmm computation for the experiment.";
        using type = bool;
        enum
        {
            flagged
        };
    };

    struct nb_runs : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--nb-runs", "-nruns"};
        std::string description = "The number of experiments to run.";
        using type = std::size_t;
        type def = 5;
        std::string input_hint = "std::size_t";
    };

    struct size : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--size", "-N"};
        std::string description = "Number of bodies.";
        using type = std::size_t;
        type def = 100;
    };

    struct thread_count : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--threads", "-t"};
        std::string description = "Maximum thread count to be used.";
        using type = std::size_t;
        type def = 1;
    };

    struct order : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--order", "-o"};
        std::string description{"Order of the approximation."};
        using type = std::size_t;
        type def = 3;
    };

    struct tree_height : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--tree-height", "-th"};
        std::string description = "The height of the tree.";
        using type = std::size_t;
        type def = 3;
        std::string input_hint = "std::size_t";
    };

    struct group_size : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--group-size", "-gs"};
        std::string description = "The size of the group (blocks) for task granularity.";
        using type = std::size_t;
        type def = 1;
        std::string input_hint = "std::size_t";
    };

    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "-dim"};
        std::string description = "Dimension of the problem.";
        using type = std::size_t;
        type def = 2;
        std::string input_hint = "std::size_t";
    };

    struct operators_to_proceed : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--operators-to-proceed", "-op"};
        std::string description =
          "Operators to proceed = \n" + std::string("- p2p = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::p2p) + "\n" + std::string("- p2m = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::p2m) + "\n" + std::string("- m2m = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::m2m) + "\n" + std::string("- m2l = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::m2l) + "\n" + std::string("- l2l = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::l2l) + "\n" + std::string("- l2p = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::l2p) + "\n" + std::string("- p2l = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::p2l) + "\n" + std::string("- m2p = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::m2p) + "\n" + std::string("- nearfield = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::nearfield) + "\n" + std::string("- farfield = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::farfield) + "\n" + std::string("- all = ") +
          std::to_string(scalfmm::algorithms::operators_to_proceed::all);
        using type = unsigned int;
        type def = scalfmm::algorithms::operators_to_proceed::all;
        std::string input_hint = "unsigned int";
    };

    struct use_float
    {
        cpp_tools::cl_parser::str_vec flags = {"--use-float"};
        std::string description = "Enable to perform the computation in single precision.";
        using type = bool;
        enum
        {
            flagged
        };
    };
    struct gauss_coeff
    {
        cpp_tools::cl_parser::str_vec flags = {"--gauss-coeff", "-gc"};
        std::string description = "Coefficient of the gaussian.";
        using type = double;
        std::string input_hint = "double"; /*!< The input hint */
        type def = type(1.0);
    };
    struct gauss_regul
    {
        cpp_tools::cl_parser::str_vec flags = {"--gauss-regul", "-gr"};
        std::string description = "Regularisation Coefficient of the gaussian on the diagonal term.";
        using type = double;
        std::string input_hint = "double"; /*!< The input hint */
        type def = type(0.0);
    };
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
    {
        os << "{ ";
        for(auto el: vec)
        {
            os << el << ' ';
        }
        os << "}";
        return os;
    }

    template<typename ParserType, typename T, typename... Args>
    auto print_parameters(ParserType const& parser, T&& value, Args&&... args) -> void
    {
        T dummy;
        std::cout << cpp_tools::colors::cyan;
        std::cout << std::boolalpha;
        std::cout << "[param] " << std::left << std::setw(24) << dummy.flags[0] << " = " << parser.template get<T>()
                  << "\n";
        if constexpr(sizeof...(args) > 0)
        {
            print_parameters(parser, std::forward<Args>(args)...);
        }
        std::cout << cpp_tools::colors::reset;
    }

}   // namespace local_args

template<typename ContainerType>
auto reset_output(ContainerType& container) -> void
{
    using container_type = ContainerType;
    using particle_type = typename container_type::value_type;
    using value_type = typename particle_type::outputs_value_type;

    for(auto& particle: container)
    {
        for(auto& entry: particle.outputs())
        {
            entry = value_type(0.);
        }
    }
}

template<typename ContainerType>
auto read_data(const std::string& filename)
{
    using container_type = ContainerType;
    using particle_type = typename container_type::value_type;
    using value_type = typename particle_type::position_value_type;
    static constexpr std::size_t dimension{particle_type::dimension};

    const bool verbose{false};

    scalfmm::io::FFmaGenericLoader<value_type, dimension> loader(filename, verbose);

    const std::size_t size{loader.getNumberOfParticles()};
    const auto box_width{loader.getBoxWidth()};
    const auto box_center{loader.getBoxCenter()};

    std::cout << cpp_tools::colors::yellow << '\n';
    std::cout << "[file][nb-particles] : " << size << '\n';
    std::cout << "[file][box-width]    : " << box_width << '\n';
    std::cout << "[file][box-center]   : " << box_center << '\n';
    std::cout << cpp_tools::colors::reset << '\n';

    auto nb_val_to_red_per_part = loader.getNbRecordPerline();
    // could be a problem for binary file (float double)
    std::vector<value_type> values_to_read(nb_val_to_red_per_part);

    container_type container(size);

    for(std::size_t idx = 0; idx < size; ++idx)
    {
        loader.fillParticle(values_to_read.data(), nb_val_to_red_per_part);

        particle_type& p = container[idx];
        std::size_t ii{0};
        for(auto& e: p.position())
        {
            e = values_to_read[ii++];
        }
        for(auto& e: p.inputs())
        {
            e = values_to_read[ii++];
        }
        p.variables(idx);
    }
    return std::make_tuple(container, box_center, box_width);
}

template<typename Container>
auto generate_data(std::size_t size, std::vector<double> const& center, double width)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using point_type = typename particle_type::position_type;
    using value_type = typename particle_type::position_value_type;

    static constexpr std::size_t dimension{particle_type::dimension};

    container_type container(size);
    point_type box_center{};

    for(std::size_t d = 0; d < dimension; ++d)
    {
        box_center[d] = static_cast<value_type>(center[d]);
    }

    value_type box_width = static_cast<value_type>(width);

    std::cout << cpp_tools::colors::yellow << '\n';
    std::cout << "[generate][nb-particles] : " << size << '\n';
    std::cout << "[generate][box-width]    : " << box_width << '\n';
    std::cout << "[generate][box-center]   : " << box_center << '\n';
    std::cout << cpp_tools::colors::reset << '\n';

    // random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::mt19937 gen(123);
    std::uniform_real_distribution<value_type> dis(-1., 1.);
    auto random_r = [&dis, &gen]() { return dis(gen); };

    // inserting particles in the container
    for(std::size_t idx = 0; idx < size; ++idx)
    {
        // particle_type p;
        particle_type& p = container[idx];
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
    }

    return std::make_tuple(container, box_center, box_width);
}

template<typename ValueType, std::size_t Dimension, typename FmmOperatorType, typename ParserType>
auto run(ParserType const& parser) -> void
{
    ////////////////////////////// SET-UP //////////////////////////////

    using value_type = ValueType;
    static constexpr std::size_t dimension = Dimension;
    using fmm_operator_type = FmmOperatorType;

    // near field
    using near_field_type = typename fmm_operator_type::near_field_type;
    using near_matrix_kernel_type = typename near_field_type::matrix_kernel_type;

    // far field
    using far_field_type = typename fmm_operator_type::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;
    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;

    // number of inputs and outputs
    static constexpr std::size_t nb_inputs_near = near_matrix_kernel_type::km;
    static constexpr std::size_t nb_outputs_near = near_matrix_kernel_type::kn;
    static constexpr std::size_t nb_inputs_far = far_matrix_kernel_type::km;
    static constexpr std::size_t nb_outputs_far = far_matrix_kernel_type::kn;

    // particles
    using particle_type = scalfmm::container::particle<value_type, dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, std::size_t>;
    using position_type = typename particle_type::position_type;
    using container_type = std::vector<particle_type>;

    // group tree
    using box_type = scalfmm::component::box<position_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using storage_type = typename interpolator_type::storage_type;
    using cell_type = scalfmm::component::cell<storage_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    // time measurement
    using duration_type = std::chrono::nanoseconds;
    using timer_type = cpp_tools::timers::timer<duration_type>;

    ////////////////////////////// PARAMETER UNPACKING //////////////////////////////

    const std::size_t size{parser.template get<local_args::size>()};
    const std::size_t order{parser.template get<local_args::order>()};
    const std::size_t tree_height{parser.template get<local_args::tree_height>()};
    const std::size_t group_size{parser.template get<local_args::group_size>()};
    const std::size_t nb_runs{parser.template get<local_args::nb_runs>()};
    const std::size_t nb_threads{parser.template get<local_args::thread_count>()};
    const std::string output_file{parser.template get<local_args::output_file>()};
    const std::string input_file{parser.template get<local_args::input_file>()};

    const bool direct_computation{parser.template exists<local_args::direct_computation>()};
    const bool fmm_computation{parser.template exists<local_args::fmm_computation>()};

    const unsigned int operators_to_proceed{parser.template get<local_args::operators_to_proceed>()};

    //////////////////////////////////// OPENMP ///////////////////////////////////

    omp_set_dynamic(0);
    omp_set_num_threads(nb_threads);

    ////////////////////////////////// EXECUTION ///////////////////////////////////

    [[maybe_unused]] timer_type timer_fmm{};
    [[maybe_unused]] timer_type timer_direct{};

    position_type box_center{};
    value_type box_width{};
    container_type container{};

    // read or generate data
    if(input_file.find(".fma") != std::string::npos || input_file.find(".bfma") != std::string::npos)
    {
        std::tie(container, box_center, box_width) = read_data<container_type>(input_file);
    }
    else
    {
        std::vector<double> center(dimension, 0.);
        value_type width{2.};
        std::tie(container, box_center, box_width) = generate_data<container_type>(size, center, width);
    }

    std::size_t N{container.size()};
    box_type box(box_width, box_center);

    // construct the fmm operator
    // construct the near field
    near_field_type near_field;
    auto& near_mk = near_field.matrix_kernel();

    far_matrix_kernel_type far_mk{};
    if constexpr(local_utils::is_gaussian_kernel_v<near_matrix_kernel_type>)
    {
        value_type coeff{value_type(parser.template get<local_args::gauss_coeff>())};
        value_type reg{value_type(parser.template get<local_args::gauss_regul>())};
        // if we deal with a gaussian kernel, we set the different parameters
        // a reference on the matrix_kernel of the near_field

        near_mk.set_coeff(coeff);
        near_mk.set_epsilon(reg);

        far_mk.set_coeff(coeff);
        far_mk.set_epsilon(reg);
    }
    // build the approximation used in the near field
    interpolator_type interpolator(far_mk, order, tree_height, box.width(0));
    far_field_type far_field(interpolator);
    //  construct the fmm operator
    fmm_operator_type fmm_operator(near_field, far_field);

    group_tree_type tree(tree_height, order, box, group_size, group_size, container);

    if(fmm_computation)
    {
        // FMM algorithm
        std::cout << cpp_tools::colors::blue;
        fmm_operator.settings(std::cout);
        std::cout << cpp_tools::colors::reset;
        for(std::size_t i = 0; i < nb_runs; ++i)
        {
            // std::cout << "\tFMM run " << i << "\n";
            tree.reset_outputs();
            tree.reset_far_field();
            timer_fmm.tic();
            scalfmm::algorithms::fmm[fmm_options](tree, fmm_operator, operators_to_proceed);
            timer_fmm.tac();
            std::cout << "\n";
        }

        value_type avg_fmm_time = timer_fmm.cumulated() / static_cast<value_type>(nb_runs);
        std::cout << "[time][fmm]     : " << avg_fmm_time / 1e9 << "\n";

        // save output values in .fma or .bfma file
        if(output_file.find(".fma") != std::string::npos || output_file.find(".bfma") != std::string::npos)
        {
            std::cout << "\n\tWrite outputs in " << output_file << std::endl;
            scalfmm::io::FFmaGenericWriter<value_type> writer(output_file);
            writer.writeDataFromTree(tree, N);
            std::cout << std::endl;
        }
    }

    if(direct_computation)
    {
        // Direct algorithm
        for(std::size_t i = 0; i < nb_runs; ++i)
        {
            // std::cout << "\tDIRECT run " << i << "\n";
            reset_output(container);
            timer_direct.tic();
            scalfmm::algorithms::full_direct(container, near_mk);
            timer_direct.tac();
        }

        value_type avg_direct_time = timer_direct.cumulated() / static_cast<value_type>(nb_runs);
        std::cout << "[time][direct]  : " << avg_direct_time / 1e9 << "\n";
    }

    if(fmm_computation && direct_computation)
    {
        // Compute relative error
        using accurater_type = scalfmm::utils::accurater<value_type>;
        accurater_type error{};

        scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                          [&container, &error](auto const& leaf)
                                          {
                                              // loop on the particles of the leaf
                                              for(auto const p_ref: leaf)
                                              {
                                                  // build a particle
                                                  const auto p = typename leaf_type::const_proxy_type(p_ref);
                                                  //
                                                  const auto& idx = std::get<0>(p.variables());

                                                  auto const& output_ref = container[idx].outputs();
                                                  auto const& output = p.outputs();

                                                  for(std::size_t i{0}; i < nb_outputs_near; ++i)
                                                  {
                                                      error.add(output_ref.at(i), output.at(i));
                                                  }
                                              }
                                          });

        std::cout << cpp_tools::colors::red;
        std::cout << '\n' << error << '\n';
        std::cout << cpp_tools::colors::reset;
    }
}

template<typename ValueType, std::size_t Dimension, typename InterpSettingsType, typename Parser>
auto select_kernel(Parser const& parser) -> void
{
    using value_type = ValueType;
    using interp_settings_type = InterpSettingsType;
    static constexpr std::size_t dimension = Dimension;

    const int kernel_choice(parser.template get<local_args::kernel>());

    switch(kernel_choice)
    {
    case 0:
    {
        std::cout << cpp_tools::colors::cyan;
        if constexpr(dimension == 1)
        {
            std::cout << "[run] matrix kernel 0) one_over_x" << std::endl;
        }
        else
        {
            std::cout << "[run] matrix kernel 0) one_over_r" << std::endl;
        }
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using matrix_kernel_type = std::conditional_t<(Dimension == 1), scalfmm::matrix_kernels::one_d::one_over_x,
                                                      scalfmm::matrix_kernels::laplace::one_over_r>;

        using far_matrix_kernel_type = matrix_kernel_type;
        using near_matrix_kernel_type = matrix_kernel_type;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
#ifdef ALL_KERNELS
    case 1:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 1) one_over_r (non-symmetric)" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_non_symmetric;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_non_symmetric;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
    case 2:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 2) one_over_r (non-homogenous)" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_non_homogenous;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_non_homogenous;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
    case 3:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 3) one_over_r (non-homogenous non-symmetric)" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_non_homogenous_non_symmetric;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_non_homogenous_non_symmetric;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
    case 4:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 4) grad_one_over_r<dimension>" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
    case 5:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 5) val_grad_one_over_r<dimension>" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
#endif
    case 6:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 6) grad_one_over_r<dimension> (optimized)" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
    case 7:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 7) val_grad_one_over_r<dimension> (optimized)" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
    case 8:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] matrix kernel 8) gaussian" << std::endl;
        std::cout << cpp_tools::colors::reset;

        // matrix kernels
        using far_matrix_kernel_type = scalfmm::matrix_kernels::gaussian<value_type>;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::gaussian<value_type>;

        // near field
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

        // far field
        using interp_settings_type = InterpSettingsType;
        using interpolation_type =
          scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, interp_settings_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

        // fmm operators
        using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

        run<value_type, dimension, fmm_operator_type>(parser);
        break;
    }
    default:
    {
        throw std::invalid_argument("Invalid choice for kernel.");
        break;
    }
    }
}

template<typename ValueType, std::size_t Dimension, typename Parser>
auto select_interp(Parser const& parser) -> void
{
    using value_type = ValueType;
    static constexpr std::size_t dimension = Dimension;

    const int interp_type(parser.template get<local_args::interp_settings>());

    switch(interp_type)
    {
    case 0:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] interpolation: 0) uniform ( dense )" << std::endl;
        std::cout << cpp_tools::colors::reset;

        using interp_settings_type = scalfmm::options::uniform_<scalfmm::options::dense_>;
        select_kernel<value_type, dimension, interp_settings_type>(parser);
        break;
    }
    case 1:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] interpolation: 1) uniform ( low rank )" << std::endl;
        std::cout << cpp_tools::colors::reset;

        using interp_settings_type = scalfmm::options::uniform_<scalfmm::options::low_rank_>;
        select_kernel<value_type, dimension, interp_settings_type>(parser);
        break;
    }
    case 2:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] interpolation: 2) uniform ( fft )" << std::endl;
        std::cout << cpp_tools::colors::reset;

        using interp_settings_type = scalfmm::options::uniform_<scalfmm::options::fft_>;
        select_kernel<value_type, dimension, interp_settings_type>(parser);
        break;
    }
    case 3:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] interpolation: 3) chebyshev ( dense )" << std::endl;
        std::cout << cpp_tools::colors::reset;

        using interp_settings_type = scalfmm::options::chebyshev_<scalfmm::options::dense_>;
        select_kernel<value_type, dimension, interp_settings_type>(parser);

        break;
    }

    case 4:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "[run] interpolation: 4) chebyshev ( low rank )" << std::endl;
        std::cout << cpp_tools::colors::reset;

        using interp_settings_type = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;
        select_kernel<value_type, dimension, interp_settings_type>(parser);
        break;
    }
    default:
    {
        throw std::invalid_argument("Invalid choice for interpolation.");
        break;
    }
    }
}

template<typename ValueType, typename ParserType>
auto select_dimension(ParserType const& parser) -> void
{
    using value_type = ValueType;

    const std::size_t dim{parser.template get<local_args::dimension>()};

    switch(dim)
    {
    case 1:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "\n[run] dimension = 1" << std::endl;
        std::cout << cpp_tools::colors::reset;

        static constexpr std::size_t dimension = 1;
        select_interp<value_type, dimension>(parser);
        break;
    }
    case 2:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "\n[run] dimension = 2" << std::endl;
        std::cout << cpp_tools::colors::reset;

        static constexpr std::size_t dimension = 2;
        select_interp<value_type, dimension>(parser);
        break;
    }
    case 3:
    {
        std::cout << cpp_tools::colors::cyan;
        std::cout << "\n[run] dimension = 3" << std::endl;
        std::cout << cpp_tools::colors::reset;

        static constexpr std::size_t dimension = 3;
        select_interp<value_type, dimension>(parser);
        break;
    }
    default:
    {
        throw std::runtime_error("Invalid choice for dimension.");
        break;
    }
    }
}

template<typename ParserType>
auto select_precision(ParserType const& parser) -> void
{
    const bool use_float{parser.template exists<local_args::use_float>()};

    if(use_float)
    {
        using value_type = float;
        select_dimension<value_type>(parser);
    }
    else
    {
        using value_type = double;
        select_dimension<value_type>(parser);
    }
}

template<typename... Args>
auto get_parser(int argc, char* argv[], Args&&... args)
{
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, std::forward<Args>(args)...);
    parser.parse(argc, argv);
    local_args::print_parameters(parser, std::forward<Args>(args)...);

    return parser;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    auto parser = get_parser(argc, argv, local_args::size{}, local_args::order{}, local_args::tree_height{},
                             local_args::group_size{}, local_args::nb_runs{}, local_args::direct_computation{},
                             local_args::fmm_computation{}, local_args::input_file{}, local_args::output_file{},
                             local_args::dimension{}, local_args::use_float{}, local_args::kernel{},
                             local_args::thread_count{}, local_args::interp_settings{},
                             local_args::operators_to_proceed{}, local_args::gauss_coeff(), local_args::gauss_regul());

    select_precision(parser);

    return 0;
}
