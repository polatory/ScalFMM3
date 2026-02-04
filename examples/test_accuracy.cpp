
#include <iostream>
#include <string>
#include <vector>

#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/gaussian.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/laplace_tools.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "scalfmm/utils/sort.hpp"

#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include "scalfmm/tools/tree_io.hpp"

#include "scalfmm/tools/tree_is_nan.hpp"

// example:
// ./examples/Release/test_accuracy --input-file ../data/unitCubeXYZQ100_sorted.bfma
// --tree-height  4 -gs 2 --order 4 --data-sorted --kernel 0 --pt --output-file test_new.fma

template<typename V, std::size_t D, typename MK, typename O>
using interpolator_alias = scalfmm::interpolation::interpolator<V, D, MK, O>;
using value_type = double;
static constexpr std::size_t dimension{3};

namespace local_args
{
    struct isSorted
    {
        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--data-sorted", "--ds"};
        std::string description = "Precise if the data are sorted by their morton index";
    };

    struct outputs_already_computed
    {
        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--outputs-already-computed", "--oac"};
        std::string description = "Precise if the exact outputs (direct computation) are already computed";
    };

    struct interpolator : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--interpolator", "-i"};
        std::string description = "The interpolation : 0 for uniform, 1 for chebyshev, 2 for barycentric.";
        using type = int;
        type def = 0;
    };

    struct matrix_kernel
    {
        cpp_tools::cl_parser::str_vec flags = {"--kernel", "-k"};
        const char* description = "Matrix kernels: \n   0) 1/r, 1) expr(-r^2/l^2) ";
        using type = int;
        type def = 0;
    };
    struct variance
    {
        cpp_tools::cl_parser::str_vec flags = {"--variance", "-var"};
        const char* description = "parameter l for kernel expr(-r^2/l^2) ";
        using type = double;
        type def = 1.0;
    };
}   // namespace local_args
template<typename Container>
auto read_data(const std::string& filename, bool outputs_already_computed = false)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using value_type1 = typename particle_type::position_value_type;
    static constexpr std::size_t dimension{particle_type::dimension};
    const bool verbose{false};

    scalfmm::io::FFmaGenericLoader<value_type1, dimension> loader(filename, verbose);
    const auto width{loader.getBoxWidth()};
    const auto center{loader.getBoxCenter()};
    const std::size_t number_of_particles{loader.getNumberOfParticles()};

    auto nb_val_to_red_per_part = loader.getNbRecordPerline();
    // could be a problem for binary file (float double)
    std::vector<value_type1> values_to_read(nb_val_to_red_per_part);

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
        if(outputs_already_computed)
        {
            for(auto& e: p.outputs())
            {
                e = values_to_read[ii++];
            }
        }
        // p.variables(values_to_read[ii++], idx, 1);
        p.variables(idx);
        container[idx] = p;
    }
    return std::make_tuple(container, center, width);
}

template<class FmmOperatorType, typename NearMatrixKernelType, typename FarMatrixKernelType>
auto run(const std::string& input_file, const std::string& output_file, const int& tree_height, const int& group_size,
         const int& order, NearMatrixKernelType& mk_near, FarMatrixKernelType& mk_far,
         const bool& outputs_already_computed = false) -> int
{
    using near_matrix_kernel_type = typename FmmOperatorType::near_field_type::matrix_kernel_type;
    using interpolator_type = typename FmmOperatorType::far_field_type::approximation_type;
    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;

    static_assert(std::is_same_v<near_matrix_kernel_type, NearMatrixKernelType>,
                  "Matrix kernel types do not match (near field)");
    static_assert(std::is_same_v<far_matrix_kernel_type, FarMatrixKernelType>,
                  "Matrix kernel types do not match (far field)");

    static constexpr std::size_t dimension{interpolator_type::dimension};
    //
    //
    bool dataSorted = false;
    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    //  The matrix kernel
    //
    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};

    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    // Open particle file
    cpp_tools::timers::timer time{};

    // ---------------------------------------
    using particle_type = scalfmm::container::particle<value_type, dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, std::size_t>;
    using container_type = std::vector<particle_type>;
    using point_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<point_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "Creating & Inserting particles ...\n" << cpp_tools::colors::reset;

    point_type box_center{};
    value_type box_width{};
    container_type container{};
    time.tic();
    std::tie(container, box_center, box_width) = read_data<container_type>(input_file, outputs_already_computed);
    time.tac();

    // const std::size_t number_of_particles = std::get<0>(container->size());
    const std::size_t number_of_particles = container.size();
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << " box width = " << box_width
              << cpp_tools::colors::reset << '\n';

    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    time.tic();
    box_type box(box_width, box_center);
    group_tree_type tree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                         static_cast<std::size_t>(group_size), container, dataSorted);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Group tree created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    time.tic();
    interpolator_type interpolator(mk_far, order, static_cast<std::size_t>(tree_height), box.width(0));
    typename FmmOperatorType::near_field_type near_field(mk_near);
    typename FmmOperatorType::far_field_type far_field(interpolator);
    FmmOperatorType fmm_operator(near_field, far_field);
    time.tac();

    std::cout << cpp_tools::colors::blue << "Fmm with kernels: " << std::endl
              << "       near " << mk_near.name() << std::endl
              << "       far  " << mk_far.name() << std::endl
              << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::yellow << "Kernel and Interp created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    auto operator_to_proceed = scalfmm::algorithms::all;
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::timit)](tree, fmm_operator, operator_to_proceed);

    if(!outputs_already_computed)
    {
        time.tic();
        scalfmm::algorithms::full_direct(container, mk_near);
        time.tac();
        std::cout << cpp_tools::colors::yellow << "Direct computation done in " << time.elapsed() << "ms\n"
                  << cpp_tools::colors::reset;
    }
    else
    {
        std::cout << cpp_tools::colors::yellow << "Outputs already computed...\n" << cpp_tools::colors::reset;
    }

    scalfmm::utils::accurater<value_type> error;

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

    std::cout << error << '\n';
    if(!output_file.empty())
    {
        std::cout << "Write outputs in " << output_file << std::endl;
        scalfmm::io::FFmaGenericWriter<double> writer(output_file);
        writer.writeDataFromTree(tree, number_of_particles);
    }

    return 0;
}

template<class OPTION_TYPE>
auto select_kernel(const int& matrix_type, const std::string& input_file, const std::string& output_file,
                   const int& tree_height, const int& group_size, const int& order, const value_type& variance,
                   const bool& outputs_already_computed = false) -> void
{
    switch(matrix_type)
    {
    // case 0:
    // {
    //     using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    //     using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
    //     //
    //     using interpolation_type = interpolator_alias<value_type, dimension, matrix_kernel_type, OPTION_TYPE>;
    //     using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
    //     matrix_kernel_type mk{};
    //     run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
    //       input_file, output_file, tree_height, group_size, order, mk, outputs_already_computed);
    //     break;
    // }
    // case 1:
    // {
    //     using matrix_kernel_type = scalfmm::matrix_kernels::gaussian<value_type>;
    //     using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
    //     //
    //     using interpolation_type = interpolator_alias<value_type, dimension, matrix_kernel_type, OPTION_TYPE>;
    //     using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
    //     matrix_kernel_type mk{};

    //     mk.set_coeff(variance);
    //     run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
    //       input_file, output_file, tree_height, group_size, order, mk, outputs_already_computed);
    //     break;
    // }
    // TEMP AG
    case 0:
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolation_type = interpolator_alias<value_type, dimension, matrix_kernel_type, OPTION_TYPE>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
        //
        matrix_kernel_type mk{};

        run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
          input_file, output_file, tree_height, group_size, order, mk, mk, outputs_already_computed);
        break;
    }
    case 1:
    {
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using interpolation_type = interpolator_alias<value_type, dimension, far_matrix_kernel_type, OPTION_TYPE>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;
        //
        near_matrix_kernel_type mk_near{};
        far_matrix_kernel_type mk_far{};

        run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
          input_file, output_file, tree_height, group_size, order, mk_near, mk_far, outputs_already_computed);
        break;
    }
    default:
        std::cout << "Kernel not implemented. value is  0) 1/r, 1) exp(-r^2/l^2),  " << std::endl;
        break;
    }
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, args::input_file(), args::output_file(), args::tree_height{},
      args::order{},   // args::thread_count{},
      args::block_size{}, args::log_file{}, args::log_level{}, local_args::isSorted{}, local_args::matrix_kernel{},
      local_args::interpolator{}, local_args::variance{}, local_args::outputs_already_computed{});
    parser.parse(argc, argv);
    // Getting command line parameters
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

    const auto output_file{parser.get<args::output_file>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    const auto order{parser.get<args::order>()};
    // const bool dataSorted(parser.exists<local_args::isSorted>());
    const bool outputs_already_computed(parser.get<local_args::outputs_already_computed>());
    const int matrix_type = parser.get<local_args::matrix_kernel>();
    const int which_interp(parser.get<local_args::interpolator>());
    const value_type variance(parser.get<local_args::variance>());

    if(which_interp == 0)
    {
        std::cout << cpp_tools::colors::blue << "Fmm interpolation: uniform (uniform fft)" << std::endl
                  << cpp_tools::colors::reset;
        using options_uniform = scalfmm::options::uniform_<scalfmm::options::fft_>;
        select_kernel<options_uniform>(matrix_type, input_file, output_file, tree_height, group_size, order, variance,
                                       outputs_already_computed);
    }
    else if(which_interp == 1)
    {
        std::cout << cpp_tools::colors::blue << "Fmm interpolation: chebyshev (chebyshev low-rank)" << std::endl
                  << cpp_tools::colors::reset;
        using options_chebyshev = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;
        select_kernel<options_chebyshev>(matrix_type, input_file, output_file, tree_height, group_size, order, variance,
                                         outputs_already_computed);
    }
    else if(which_interp == 2)
    {
        std::cout << cpp_tools::colors::blue << "Fmm interpolation: barycentric (barycentric dense)" << std::endl
                  << cpp_tools::colors::reset;
        using options_barycentric = scalfmm::options::barycentric_<scalfmm::options::dense_>;
        select_kernel<options_barycentric>(matrix_type, input_file, output_file, tree_height, group_size, order,
                                           variance, outputs_already_computed);
    }
    else if(which_interp == 3)
    {
        std::cout << cpp_tools::colors::blue << "Fmm interpolation: barycentric (barycentric low-rank)" << std::endl
                  << cpp_tools::colors::reset;
        using options_barycentric = scalfmm::options::barycentric_<scalfmm::options::low_rank_>;
        select_kernel<options_barycentric>(matrix_type, input_file, output_file, tree_height, group_size, order,
                                           variance, outputs_already_computed);
    }

    return 0;
}
