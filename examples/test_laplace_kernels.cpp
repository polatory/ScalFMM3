
#include <iostream>
#include <string>
#include <vector>

#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/laplace_tools.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "scalfmm/utils/sort.hpp"

#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

// example:
// ./examples/RelWithDebInfo/test_laplace_kernels --input-file ../data/unitCubeXYZQ100_sorted.bfma
// --tree-height  4 -gs 2 --order 4 --data-sorted --kernel 0 --pt --output-file test_new.fma
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

struct interpolator : cpp_tools::cl_parser::required_tag
{
    cpp_tools::cl_parser::str_vec flags = {"--interpolator", "-interp"};
    std::string description = "The interpolation : 0 for uniform, 1 for chebyshev.";
    using type = int;
    type def = 0;
};

template<class FMM_OPERATOR_TYPE>
auto run(const std::string& input_file, const std::string& output_file, const int& tree_height, const int& group_size,
         const int& order, const bool postreat, const bool dataSorted) -> int
{
    using value_type = double;
    using near_matrix_kernel_type = typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;
    using interpolator_type = typename FMM_OPERATOR_TYPE::far_field_type::approximation_type;
    static constexpr std::size_t dimension{interpolator_type::dimension};

    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
    //
    //
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
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "Creating & Inserting particles ...\n" << cpp_tools::colors::reset;

    scalfmm::container::point<value_type, dimension> box_center{};
    value_type box_width{};

    time.tic();
    container_type* container{};
    laplace::read_data<3>(input_file, container, box_center, box_width);
    time.tac();

    // const std::size_t number_of_particles = std::get<0>(container->size());
    const std::size_t number_of_particles = container->size();
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << " box width = " << box_width
              << cpp_tools::colors::reset << '\n';

    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    time.tic();
    box_type box(box_width, box_center);
    group_tree_type gtree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                          static_cast<std::size_t>(group_size), *container, dataSorted);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Group tree created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;
    delete container;
    time.tic();
    far_matrix_kernel_type mk_far{};
    interpolator_type interpolator(mk_far, order, static_cast<std::size_t>(tree_height), box.width(0));
    near_matrix_kernel_type mk_near{};
    typename FMM_OPERATOR_TYPE::near_field_type near_field(mk_near);
    typename FMM_OPERATOR_TYPE::far_field_type far_field(interpolator);
    FMM_OPERATOR_TYPE fmm_operator(near_field, far_field);
    time.tac();
    std::cout << cpp_tools::colors::blue << "Fmm with kernels: " << std::endl
              << "       near " << mk_near.name() << std::endl
              << "       far  " << mk_far.name() << std::endl
              << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::yellow << "Kernel and Interp created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    auto operator_to_proceed = scalfmm::algorithms::all;
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::timit)](gtree, fmm_operator, operator_to_proceed);

    if(postreat)
    {   // Doesn't work because gtree is not a container
        //   laplace::post_traitement(mk_near, &gtree);
    }

    if(!output_file.empty())
    {
        std::cout << "Write outputs in " << output_file << std::endl;
        scalfmm::io::FFmaGenericWriter<double> writer(output_file);
        writer.writeDataFromTree(gtree, number_of_particles);
    }

    return 0;
}

template<typename V, std::size_t D, typename MK, typename O>
using interpolator_alias = scalfmm::interpolation::interpolator<V, D, MK, O>;

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    static constexpr std::size_t dimension{3};
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, args::input_file(), args::output_file(), args::tree_height{},
      args::order{},   // args::thread_count{},
      args::block_size{}, args::log_file{}, args::log_level{}, isSorted{}, laplace::args::matrix_kernel{},
      laplace::args::post_traitement{}, interpolator{});
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
    const bool postreat(parser.exists<laplace::args::post_traitement>());
    const bool dataSorted(parser.exists<isSorted>());
    const int matrix_type = parser.get<laplace::args::matrix_kernel>();
    const int which_interp(parser.get<interpolator>());

    if(which_interp == 0)
    {
        using options_uniform = scalfmm::options::uniform_<scalfmm::options::fft_>;
	std::cout << cpp_tools::colors::blue << "Fmm interpolation: uniform" << std::endl
		<< cpp_tools::colors::reset;

        switch(matrix_type)
        {
        case 0:
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
            using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
            //
            using interpolation_type = interpolator_alias<double, dimension, matrix_kernel_type, options_uniform>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }
        break;
        case 1:
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
            using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
            //
            using interpolation_type = interpolator_alias<double, dimension, matrix_kernel_type, options_uniform>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }
        break;
        case 2:
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
            using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
            //
            using interpolation_type = interpolator_alias<double, dimension, matrix_kernel_type, options_uniform>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }

        break;
        case 3:
        {
            using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
            using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

            using interpolation_type = interpolator_alias<double, dimension, far_matrix_kernel_type, options_uniform>;
            using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
            // true beacause we compute the gradient of the interpolator
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }
        break;
        case 4:
        {
            using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
            using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

            using interpolation_type = interpolator_alias<double, dimension, far_matrix_kernel_type, options_uniform>;
            using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
            // true beacause we compute the gradient of the interpolator
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }

        break;
        default:
            std::cout << "Kernel not implemented. value is  0) 1/r, 1) grad(1/r),  "
                         "2) p & grad(1/r) "
                      << std::endl
                      << "3) optimized grad(1/r)  4) optimized p & grad(1/r) .  " << std::endl;
            break;
        }
    }
    else if(which_interp == 1)
    {
      std::cout << cpp_tools::colors::blue << "Fmm interpolation: chebyshev" << std::endl
		<< cpp_tools::colors::reset;
        using options_chebyshev = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;
        switch(matrix_type)
        {
        case 0:
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
            using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
            //
            using interpolation_type = interpolator_alias<double, dimension, matrix_kernel_type, options_chebyshev>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }
        break;
        case 1:
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
            using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
            //
            using interpolation_type = interpolator_alias<double, dimension, matrix_kernel_type, options_chebyshev>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }
        break;
        case 2:
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
            using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
            //
            using interpolation_type = interpolator_alias<double, dimension, matrix_kernel_type, options_chebyshev>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }

        break;
        case 3:
        {
            using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<dimension>;
            using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

            using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
            using interpolation_type = interpolator_alias<double, dimension, far_matrix_kernel_type, options_chebyshev>;
            // true because we compute the gradient of the interpolator
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }
        break;
        case 4:
        {
            using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
            using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

            using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
            using interpolation_type = interpolator_alias<double, dimension, far_matrix_kernel_type, options_chebyshev>;
            // true because we compute the gradient of the interpolator
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;

            run<scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
              input_file, output_file, tree_height, group_size, order, postreat, dataSorted);
        }
        break;
        default:
            std::cout << "Kernel not implemented. value is  0) 1/r, 1) grad(1/r),  "
                         "2) p & grad(1/r) "
                      << std::endl
                      << "3) optimized grad(1/r)  4) optimized p & grad(1/r) .  " << std::endl;
            break;
        }
    }

    return 0;
}
