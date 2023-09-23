#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/chebyshev.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/accurater.hpp"
#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

namespace local_args
{
    struct tree_height : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--tree-height", "-th"};
        std::string description = "Tree height (or initial height in case of an adaptive tree).";
        using type = std::size_t;
        type def = 2;
    };
    struct order : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--order", "-o"};
        std::string description = "Precision setting.";
        using type = std::size_t;
        type def = 3;
    };
    struct thread_count
    {
        cpp_tools::cl_parser::str_vec flags = {"--threads", "-t"};
        std::string description = "Maximum thread count to be used.";
        using type = std::size_t;
        type def = 1;
    };
    struct input_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };
    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
    };
    struct block_size
    {
        cpp_tools::cl_parser::str_vec flags = {"--group-size", "-gs"};
        std::string description = "Group tree chunk size.";
        using type = int;
        type def = 250;
    };
    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n   2 for dimension 2, 3 for dimension 3";
        using type = std::size_t;
        type def = 1;
    };
}   // namespace local_args

template<std::size_t Dimension, typename ContainerType, typename PointType, typename ValueType>
void read_data(const std::string& filename, ContainerType*& container, PointType& Centre, ValueType& width)
{
    //  std::cout << "READ DATA " << std::endl << std::flush;
    using particle_type = typename ContainerType::particle_type;
    bool verbose = true;

    scalfmm::io::FFmaGenericLoader<ValueType, Dimension> loader(filename, verbose);

    const int number_of_particles = loader.getNumberOfParticles();
    width = loader.getBoxWidth();
    Centre = loader.getBoxCenter();
    //    std::size_t i{0};
    //    for(auto& e: Centre)
    //    {
    //        e = Centre_3D[i++];
    //    }
    auto nb_val_to_red_per_part = loader.getNbRecordPerline();

    double* values_to_read = new double[nb_val_to_red_per_part]{0};
    container = new ContainerType(number_of_particles);
    std::cout << "number_of_particles " << number_of_particles << std::endl;
    for(std::size_t idx = 0; idx < number_of_particles; ++idx)
    {
        loader.fillParticle(values_to_read, nb_val_to_red_per_part);
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
        p.variables(values_to_read[ii++], idx);
        container->insert_particle(idx, p);
    }
}

template<typename Tree_T, typename Container_T, typename Value_T>
auto inline check_results(const Tree_T& tree, const Container_T& reference, const Value_T& eps) -> bool
{
    scalfmm::utils::accurater<Value_T> error;

    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&reference, &error](auto const& leaf)
                                      {
                                          const auto nb_elt = leaf.size();
                                          for(std::size_t i = 0; i < nb_elt; ++i)
                                          {
                                              const auto& p = leaf.particle(i);
                                              std::cout << p << std::endl;
                                              const auto& idx = std::get<0>(p.variables());
                                              error.add(std::get<0>(p.outputs()),
                                                        std::get<1>(reference.variables(idx)));
                                          }
                                      });
    bool ok = (error.get_relative_l2_norm() < eps);
    std::cout << "Error " << error << std::endl;
    return ok;
}
template<std::size_t Dimension, typename FmmOperatorType, typename... Parameters>
auto run(cpp_tools::cl_parser::parser<Parameters...> const& parser) -> int
{
    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;
    // timer
    cpp_tools::timers::timer time{};

    using value_type = double;
    // ---------------------------------------
    using near_matrix_kernel_type = typename FmmOperatorType::near_field_type::matrix_kernel_type;
    using far_field_type = typename FmmOperatorType::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;
    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
    // ---------------------------------------
    //  The matrix kernel
    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};
    // ---------------------------------------
    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, value_type, std::size_t>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    // ---------------------------------------
    // parameters
    const auto tree_height{parser.template get<local_args::tree_height>()};
    const auto group_size{parser.template get<local_args::block_size>()};
    const auto order{parser.template get<local_args::order>()};
    const std::string input_file(parser.template get<local_args::input_file>());
    const std::string output_file(parser.template get<local_args::output_file>());

    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    // ---------------------------------------
    // Open particle file
    std::cout << cpp_tools::colors::green << "Creating & Inserting particles ...\n" << cpp_tools::colors::reset;

    scalfmm::container::point<value_type, Dimension> box_center{};
    value_type box_width{};
    container_type* container{};

    if(input_file.empty())
    {
        std::cerr << cpp_tools::colors::red << "input file is empty !\n";
        return 0;
    }
    else
    {
        time.tic();
        read_data<Dimension>(input_file, container, box_center, box_width);
        time.tac();
    }
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << " box width = " << box_width
              << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    // ---------------------------------------
    // creating group_tree
    time.tic();
    box_type box(box_width, box_center);
    group_tree_type tree(tree_height, order, box, group_size, group_size, *container);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Group tree created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    // ---------------------------------------
    // creating interpolator
    time.tic();
    far_matrix_kernel_type mk_far{};
    interpolator_type interpolator(mk_far, order, static_cast<std::size_t>(tree_height), box.width(0));
    std::cout << cpp_tools::colors::on_blue << interpolator.roots() << cpp_tools::colors::reset << '\n';
    near_matrix_kernel_type mk_near{};
    typename FmmOperatorType::near_field_type near_field(mk_near);
    typename FmmOperatorType::far_field_type far_field(interpolator);
    FmmOperatorType fmm_operator(near_field, far_field);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Kernel and Interp created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    // ---------------------------------------
    // start algorithm for step 0
    auto operator_to_proceed = scalfmm::algorithms::all;
    scalfmm::algorithms::sequential::sequential(tree, fmm_operator, operator_to_proceed);

    // ---------------------------------------
    // write output fma
    if(!output_file.empty())
    {
        std::cout << "Write outputs in " << output_file << std::endl;
        scalfmm::io::FFmaGenericWriter<double> writer(output_file);
        writer.writeDataFromTree(tree, container->size());
    }

    value_type eps = std::pow(10.0, 1 - order);
    bool works = check_results(tree, *container, eps);

    delete container;

    return works;
}

template<typename V, std::size_t D, typename MK>
using interpolator_alias =
  scalfmm::interpolation::interpolator<V, D, MK, scalfmm::options::chebyshev_<>>;

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, local_args::tree_height{}, local_args::order{}, local_args::input_file{},
      local_args::output_file{}, local_args::block_size{}, local_args::dimension{});
    parser.parse(argc, argv);
    // Getting command line parameters

    const std::size_t dimension = parser.get<local_args::dimension>();


    switch(dimension)
    {
    case 1:
    {
        constexpr std::size_t dim = 1;
        using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        using interpolator_type = interpolator_alias<double, dim, matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(parser);
        break;
    }
    case 2:
    {
        constexpr std::size_t dim = 2;
        using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        using interpolator_type = interpolator_alias<double, dim, matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(parser);
        break;
    }
    case 3:
    {
        constexpr std::size_t dim = 3;
        using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        using interpolator_type = interpolator_alias<double, dim, matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
        run<dim, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(parser);
        break;
    }
    default:
        std::cout << "check  1/r^2 Kernel for dimension 2 and 3. Value is \n"
                  << "          0 for dimension 2 "
                  << "          1 for dimension 3 " << std::endl;
        break;
    }
}
