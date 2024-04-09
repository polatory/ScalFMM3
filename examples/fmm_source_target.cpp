// @FUSE_OMP
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include <scalfmm/container/iterator.hpp>
//
//
// #include "scalfmm/meta/type_pack.hpp"
//
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
//
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/lists/lists.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
//
// Tree
#include "scalfmm/tools/tree_io.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"

//
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/laplace_tools.hpp"

#include "scalfmm/utils/compare_results.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "scalfmm/utils/source_target.hpp"

#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

using namespace scalfmm::io;
/// \file  fmm_source_target.cpp
//!
//! \brief 
//! add flag -DST_USE_OMP to compile an openmp algorithm
//!
//! \code
//! USAGE:
//! ./examples/Release/fmm_source_target [--help] --input-source-file value --input-target-file value [--output-file
//! value] [--kernel value] --dimension value
//!
//! DESCRIPTION:
//!
//!    --help, -h
//!        Display this help message
//!
//!    --input-source-file, -isf value
//!        Input filename (.fma or .bfma).
//!
//!    --input-target-file, -itf value
//!        Input filename (.fma or .bfma).
//!
//!    --output-file, -fout value
//!        Output particle file (with extension .fma (ascii) or bfma (binary).
//!
//!    --kernel, --k value
//!        Matrix kernels:
//!        0 1/r, 1) grad(1/r), 2) p & grad(1/r) 3) shift grad,
//!        4) 1/r^2  5) ln in 2d
//!
//!    --dimension, -d value
//!        Dimension :
//!        -  1 <dimension <4
//!  ./tools/direct_source_target --input-source-file test_source.fma --input-target-file test_target.fma --dimension 2
//!  --kernel 0
//! ./examples/Release/fmm_source_target  --input-source-file test_source.fma --input-target-file test_target.fma
//! --dimension 3 --kernel 0 --check
//!
//!  /examples/Release/fmm_source_target  --input-source-file sphere_source.fma --input-target-file sphere_target.fma
//!  --dimension 3 --kernel 0 --check
//!
namespace local_args
{
    struct input_source_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-source-file", "-isf"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };
    struct input_target_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-target-file", "-itf"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };
    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
    };
    struct newmatrix_kernel : public laplace::args::matrix_kernel
    {
        std::string description = "Matrix kernels: \n   0 1/r, 1) grad(1/r), 2) p & grad(1/r) 3) shift grad,\n"
                                  "4) 1/r^2  5) ln in 2d ";
    };
    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "-d"};
        std::string description = "Dimension : \n  -  1 <dimension <4";
        using type = int;
        type def = 1;
    };
    struct check
    {
        cpp_tools::cl_parser::str_vec flags = {"--check"};
        std::string description = "Check with direct computation";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
}   // namespace local_args

template<typename Tree>
auto print_leaves(Tree const& tree) -> void
{
    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&tree](auto& leaf) { scalfmm::io::print_leaf(leaf); });
}
template<typename Tree, typename Container>
auto check_output(Container const& part, Tree const& tree)
{
    scalfmm::utils::accurater<
      typename scalfmm::container::particle_traits<typename Container::value_type>::outputs_value_type>
      error;
    static constexpr std::size_t nb_out = Container::value_type::outputs_size;

    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&part, &error](auto& leaf)
                                      {
                                          for(auto const p_tuple_ref: leaf)
                                          {
                                              const auto& p = typename Tree::leaf_type::const_proxy_type(p_tuple_ref);
                                              const auto& idx = std::get<0>(p.variables());
                                              auto& output = p.outputs();
                                              const auto output_ref = part.at(idx).outputs();
                                              for(int i{0}; i < nb_out; ++i)
                                              {
                                                  error.add(output_ref.at(i), output.at(i));
                                              }
                                          }
                                      });

    return error;
}

template<typename Container>
auto read_data(const std::string& filename)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using value_type = typename particle_type::position_value_type;
    static constexpr std::size_t dimension{particle_type::dimension};
    const bool verbose{false};

    scalfmm::io::FFmaGenericLoader<value_type, dimension> loader(filename, verbose);
    const auto width{loader.getBoxWidth()};
    const auto center{loader.getBoxCenter()};
    const std::size_t number_of_particles{loader.getNumberOfParticles()};

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
        p.variables(idx);
        container[idx] = p;
    }
    return std::make_tuple(container, center, width);
}

template<int dimension, typename value_type, class fmm_operators_type>
auto fmm_run(const std::string& input_source_file, const std::string& input_target_file, const int& tree_height,
             const int& group_size, const int& order, bool check_direct, const std::string& output_file) -> int
{
    bool display_container = false;
    bool display_tree = true;
    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    //  The matrix kernel
    using near_matrix_kernel_type = typename fmm_operators_type::near_field_type::matrix_kernel_type;
    using far_field_type = typename fmm_operators_type::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;

    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
    static constexpr std::size_t nb_inputs{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs{far_matrix_kernel_type::kn};
    //
    // Open particles files

    cpp_tools::timers::timer<std::chrono::minutes> time{};

    constexpr int zeros{1};   // should be zero
    using point_type = scalfmm::container::point<value_type, dimension>;
    using box_type = scalfmm::component::box<point_type>;
    //
    using particle_source_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, zeros, std::size_t>;
    using particle_target_type =
      scalfmm::container::particle<value_type, dimension, value_type, zeros, value_type, nb_outputs, std::size_t>;

    // Construct the container of particles
    using container_source_type = std::vector<particle_source_type>;
    using container_target_type = std::vector<particle_target_type>;
    //
    using leaf_source_type = scalfmm::component::leaf_view<particle_source_type>;
    using leaf_target_type = scalfmm::component::leaf_view<particle_target_type>;
    //

    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using tree_source_type = scalfmm::component::group_tree_view<cell_type, leaf_source_type, box_type>;
    using tree_target_type = scalfmm::component::group_tree_view<cell_type, leaf_target_type, box_type>;
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << cpp_tools::colors::green << "Creating & Inserting ...\n" << cpp_tools::colors::reset;
    //
    time.tic();
    point_type box_center_source{};
    value_type box_width_source{};
    container_source_type container_source{};
    std::tie(container_source, box_center_source, box_width_source) =
      read_data<container_source_type>(input_source_file);
    box_type box_source(box_width_source, box_center_source);

    point_type box_center_target{};
    value_type box_width_target{};
    container_target_type container_target{};
    std::tie(container_target, box_center_target, box_width_target) =
      read_data<container_target_type>(input_target_file);
    box_type box_target(box_width_target, box_center_target);
    time.tac();
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;

    if(display_container)
    {
        std::cout << "Read source particles \n";
        std::cout << "box_source " << box_source << std::endl;
        scalfmm::io::print("container_source\\n", container_source);

        std::cout << cpp_tools::colors::green << "Box center = " << box_center_source
                  << " box width = " << box_width_source << cpp_tools::colors::reset << '\n';
        std::cout << "Read target particles \n";
        std::cout << "box_target " << box_target << std::endl;
        scalfmm::io::print("container_target\\n", container_target);

        std::cout << cpp_tools::colors::green << "Box center = " << box_center_target
                  << " box width = " << box_width_target << cpp_tools::colors::reset << '\n';
    }
    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << " s\n"
              << cpp_tools::colors::reset;

    auto box = scalfmm::utils::bounding_box(box_source, box_target);
    std::cout << "bounding_box " << box << std::endl;
    // auto container_all = scalfmm::utils::merge(container_source, container_target) ;

    // build trees
    bool sorted = false;
    tree_source_type tree_source(tree_height, order, box, group_size, group_size, container_source, sorted);

    tree_target_type tree_target(tree_height, order, box, group_size, group_size, container_target, sorted);

    if(display_tree)
    {
        std::cout << "Tree source\n";
        print_leaves(tree_source);
        std::cout << "Tree target\n";
        print_leaves(tree_target);
    }
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //              Compute source-target interaction though FMM
    //
    /////////////////////////////////////////////////////////////////////////////////////
    auto box_width = box.width(0);
    // Far field
    // far_matrix_kernel_type mk_far{};
    interpolator_type interpolator(order, tree_height, box_width);
    typename fmm_operators_type::far_field_type far_field(interpolator);
    // Near field with no mutual interactions
    bool mutual(false);
    typename fmm_operators_type::near_field_type near_field(mutual);
    auto& mk_near = near_field.matrix_kernel();
    auto& mk_far = far_field.approximation().matrix_kernel();
    //
    std::cout << cpp_tools::colors::blue << "Fmm with kernels: " << std::endl
              << "       near " << mk_near.name() << std::endl
              << "       far  " << mk_far.name() << std::endl;

    fmm_operators_type fmm_operator(near_field, far_field);
    auto neighbour_separation = fmm_operator.near_field().separation_criterion();
    //
    scalfmm::list::sequential::build_interaction_lists(tree_source, tree_target, neighbour_separation, mutual);
    auto operator_to_proceed = scalfmm::algorithms::all;
#ifndef ST_USE_OMP  
    scalfmm::algorithms::omp::task_dep(tree_source, tree_target, fmm_operator, operator_to_proceed);
#else
     scalfmm::algorithms::sequential::sequential(tree_source, tree_target, fmm_operator, operator_to_proceed);
#endif
    std::cout << "\n" << cpp_tools::colors::reset;

    /////////////////////////////////////////////////////////////////////////////////////
    //
    //              Check with the direct computation
    //
    /////////////////////////////////////////////////////////////////////////////////////
    if(check_direct)
    {
        std::cout << cpp_tools::colors::green << "full interaction computation  with kernel: " << mk_near.name()
                  << std::endl
                  << cpp_tools::colors::reset;

        time.tic();

        scalfmm::algorithms::full_direct(container_source, container_target, mk_near);
        time.tac();
        std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
        std::cout << cpp_tools::colors::yellow << "Computation done in " << time.elapsed() << " min\n"
                  << cpp_tools::colors::reset;
        // check the two containers
        // std::cout << "Final target container\n";
        // std::cout << container_target << std::endl;
        // Compare with the FMM computation
        auto error{check_output(container_target, tree_target).get_relative_l2_norm()};
        std::cout << cpp_tools::colors::magenta << "relative L2 error: " << error << '\n' << cpp_tools::colors::reset;
    }
    if(!output_file.empty())
    {
        std::cout << "Write outputs in " << output_file << std::endl;
        scalfmm::io::FFmaGenericWriter<value_type> writer(output_file);
        // auto & container = container_target ;
        // writer.writeDataFrom(container, container.size(), box.center(), box.width(0));
        writer.writeDataFromTree(tree_target, container_target.size());
    }

    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    using value_type = double;

    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, local_args::input_source_file(), local_args::input_target_file(),
      args::tree_height{}, args::block_size{}, args::order{}, local_args::output_file(), local_args::newmatrix_kernel{},
      local_args::dimension{}, local_args::check{});
    parser.parse(argc, argv);
    const std::string input_source_file{parser.get<local_args::input_source_file>()};
    const std::string input_target_file{parser.get<local_args::input_target_file>()};
    if(!input_source_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input source file : " << input_source_file
                  << cpp_tools::colors::reset << '\n';
    }
    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';

    const int group_size{parser.get<args::block_size>()};
    const auto order{parser.get<args::order>()};

    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';
    const auto output_file{parser.get<local_args::output_file>()};
    bool check_direct{parser.exists<local_args::check>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    const int matrix_type = parser.get<local_args::newmatrix_kernel>();
    const int dimension = parser.get<local_args::dimension>();

    //
    switch(matrix_type)
    {
    case 0:
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_matrix_kernel_type = far_matrix_kernel_type;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using options = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;

        if(dimension == 1)
        {
            using interpolation_type =
              scalfmm::interpolation::interpolator<value_type, 1, far_matrix_kernel_type, options>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

            using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;
            fmm_run<1, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
                                                       order, check_direct, output_file);
        }
        else if(dimension == 2)
        {
            using interpolation_type =
              scalfmm::interpolation::interpolator<value_type, 2, far_matrix_kernel_type, options>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

            using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;
            fmm_run<2, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
                                                       order, check_direct, output_file);
        }
        else
        {
            using interpolation_type =
              scalfmm::interpolation::interpolator<value_type, 3, far_matrix_kernel_type, options>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

            using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;
            fmm_run<3, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
                                                       order, check_direct, output_file);
        }
        break;
    // case 1:
    //     using options_case1 = scalfmm::options::uniform_<scalfmm::options::fft_>;
    //     using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

    //     if(dimension == 1)
    //     {
    //         using interpolation_type =
    //           scalfmm::interpolation::interpolator<value_type, 1, far_matrix_kernel_type, options_case1>;
    //         using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;
    //         using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;
    //         fmm_run<1, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
    //                                                    order, check_direct, output_file);
    //     }
    //     else if(dimension == 2)
    //     {
    //         using interpolation_type =
    //           scalfmm::interpolation::interpolator<value_type, 2, far_matrix_kernel_type, options>;
    //         using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;
    //         using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<2>;
    //         using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;

    //         using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    //         fmm_run<2, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
    //                                                    order, check_direct, output_file);
    //     }
    //     else
    //     {
    //         using interpolation_type =
    //           scalfmm::interpolation::interpolator<value_type, 3, far_matrix_kernel_type, options>;
    //         using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;
    //         using near_field_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<3>;

    //         using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    //         fmm_run<3, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
    //                                                    order, check_direct, output_file);
    //     }
    //     break;
    // case 2:
    //     fmm_run<3, value_type, scalfmm::matrix_kernels::laplace::val_grad_one_over_r<3>>(input_file, output_file,
    //     postreat); break;
    // case 3:
    //     fmm_run<3, value_type, scalfmm::matrix_kernels::laplace::like_mrhs>(input_file, output_file, postreat);
    //     break;
    // case 4:
    //     test_one_over_r2<value_type>(dimension, input_file, output_file, postreat);
    //     break;
    // case 5:
    //     fmm_run<2, value_type, scalfmm::matrix_kernels::laplace::ln_2d>(input_file, output_file, postreat);
    //     break;
    default:
        std::cout << "Kernel not implemented. values are\n Laplace kernels: 0) 1/r, 1) grad(1/r),"
	  //                  << " 2) p + grad(1/r) 3) like_mrhs." << std::endl
	  //                  << "Scalar kernels 4) 1/r^2 5) ln in 2d" 
		  << std::endl;
    }
}
