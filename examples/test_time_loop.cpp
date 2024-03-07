#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/vtk_writer.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf.hpp_view"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/sort.hpp"
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

//#define USE_CATALYST

#ifdef USE_CATALYST
#include "catalyst_related.hpp"
#endif

// example:
// ./examples/RelWithDebInfo/test-time-loop --tree-height  4 -gs 2 --order 4 --dimension 2...
// env PARAVIEW_LOG_CATALYST_VERBOSITY=INFO ./examples/RelWithDebInfo/test-time-loop -th 4 -o 4 -gs 1 --d 2 --ts 10 --delta 0.1 --catalyst 1 --input-file /home/p13ro/dev/cpp/gitlab/ScalFMM/experimental/data/debug/circle2d_r3.fma
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
    struct visu_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--visu-file", "-vf"};
        std::string description = "Output VTK file.";
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
    struct time_steps
    {
        cpp_tools::cl_parser::str_vec flags = {"--time-steps", "--ts"};
        std::string description = "Number of time steps";
        using type = std::size_t;
        type def = 1;
    };
    struct delta
    {
        cpp_tools::cl_parser::str_vec flags = {"--delta"};
        std::string description = "Delta to apply on forces";
        using type = double;
        type def = 0.1;
    };
    struct catalyst
    {
        cpp_tools::cl_parser::str_vec flags = {"--catalyst"};
        std::string description = "Enable catalist";
        using type = bool;
        type def = false;
    };
    struct catalyst_script
    {
        cpp_tools::cl_parser::str_vec flags = {"--catalyst-script"};
        std::string description = "Catalyst script.";
        using type = std::string;
    };
}   // namespace local_args


template<std::size_t Dimension, typename ContainerType, typename ValueType>
void read_data(const std::string& filename, ContainerType*& container,
               scalfmm::container::point<ValueType, Dimension>& Centre, ValueType& width)
{
    using particle_type = typename ContainerType::value_type;
    bool verbose = true;

    scalfmm::tools::FFmaGenericLoader<ValueType, Dimension> loader(filename, verbose);

    const int number_of_particles = loader.getNumberOfParticles();
    width = loader.getBoxWidth();
    Centre = loader.getBoxCenter();
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
            e = ValueType(1.0);
        }
        // p.variables(values_to_read[ii++], idx);
        container->insert_particle(idx, p);
    }
}

template<typename GroupTree, typename ValueType>
auto update_particles(GroupTree& tree, ValueType delta) -> void
{
    scalfmm::component::for_each_leaf(tree.begin(), tree.end(), [delta](auto& leaf)
            {
                auto& particles{leaf.particles()};
                scalfmm::container::point<ValueType, GroupTree::dimension> force{};

                for(std::size_t i{0}; i<particles.size(); ++i)
                {
                    auto proxy = particles.at(i);
                    force = -delta*proxy.position();
                    proxy.position() += force;
                }
            });
}

template<typename GroupTree>
auto get_new_box(GroupTree const& tree) -> typename GroupTree::box_type
{
    using value_type = typename GroupTree::leaf_type::particle_type::position_type::value_type;
    using box_type = typename GroupTree::box_type;
    auto const old_center = tree.box().center();

    std::vector<value_type> max(GroupTree::dimension, 0);
    std::vector<value_type> min(GroupTree::dimension, 0);

    scalfmm::component::for_each_leaf(tree.begin(), tree.end(), [&min, &max, &old_center](auto const& leaf)
            {
                auto& particles{leaf.particles()};
                for(std::size_t i{0}; i<particles.size(); ++i)
                {
                    const auto position = particles.at(i).position();
                    //std::cout << particles.at(i).inputs(0) << '\n';
                    for(std::size_t d{0}; d<GroupTree::dimension; ++d)
                    {
                        min.at(d) = std::min(position.at(d), min.at(d));
                        max.at(d) = std::max(position.at(d), max.at(d));
                        //std::cout << particles.at(i).outputs(d) << ' ';
                    }
                    //std::cout << '\n';
                }
            });
    value_type width{0};
    for(std::size_t i{0}; i < GroupTree::dimension; ++i)
    {
        width = std::max(std::abs(min.at(i)), max.at(i));
    }

    return box_type(width + std::numeric_limits<value_type>::epsilon(), old_center);
}

template<std::size_t Dimension, typename FmmOperatorType, typename... Parameters>
auto run(cpp_tools::cl_parser::parser<Parameters...> const& parser) -> int
{
    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;
    // timer
    cpp_tools::timers::timer time{};

    static constexpr auto dimension{Dimension};

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
    static constexpr std::size_t nb_inputs_far{far_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_far{far_matrix_kernel_type::kn};
    // ---------------------------------------
    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, std::size_t>;
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
    const auto n_time_steps = parser.template get<local_args::time_steps>();
    const std::string input_file(parser.template get<local_args::input_file>());
    const std::string output_file(parser.template get<local_args::output_file>());
    const std::string visu_file(parser.template get<local_args::visu_file>());
    const bool catalyst_enable(parser.template get<local_args::catalyst>());
    const std::string catalyst_script(parser.template get<local_args::catalyst_script>());
    const auto delta = parser.template get<local_args::delta>();

    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Time Steps : " << n_time_steps << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Delta forces : " << delta << cpp_tools::colors::reset << '\n';

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
        read_data(input_file, container, box_center, box_width);
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
    near_matrix_kernel_type mk_near{};
    typename FmmOperatorType::near_field_type near_field(mk_near);
    typename FmmOperatorType::far_field_type far_field(interpolator);
    FmmOperatorType fmm_operator(near_field, far_field);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Kernel and Interp created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

#ifdef USE_CATALYST
    if(catalyst_enable)
    {
        std::cout << cpp_tools::colors::on_blue << "Initializing catalyst..." << cpp_tools::colors::reset << '\n';
        catalyst_adaptor::initialize(catalyst_script);
    }
#endif
    // ---------------------------------------
    // start algorithm for step 0
    auto operator_to_proceed = scalfmm::algorithms::all;
    scalfmm::algorithms::sequential::sequential(tree, fmm_operator, operator_to_proceed);

    // ---------------------------------------
    // time loop
    for(std::size_t steps{1}; steps < n_time_steps; ++steps)
    {
        // ---
        // update particles
        update_particles(tree, delta);
        // ---
        // visu related post treatment
        if(!visu_file.empty())
        {
            scalfmm::tools::io::exportVTKxml(std::to_string(steps-1)+visu_file, tree, container->size());
        }

#ifdef USE_CATALYST
        if(catalyst_enable)
        {
            catalyst_adaptor::execute(steps-1, tree);
        }
#endif
        // ---
        // compute the corners and set new box
        auto const new_box = get_new_box(tree);
        std::cout << cpp_tools::colors::on_green << "New box width is " << new_box.width(0) << cpp_tools::colors::reset << '\n';
        // ---
        // compute new morton indices

        // ---
        // move particles according to new morton indices
        // ---
        // redistribute particle in leaves
        // ---
        // build tree levels
        // ---
        // start algorithm at step : compute forces
        // ---
    }

    // ---------------------------------------
    // last step update
    // ---
    // update particles
    // ---
    update_particles(tree, delta);

    // ---
    // visu related post treatment
    if(!visu_file.empty())
    {
        scalfmm::tools::io::exportVTKxml(std::to_string(n_time_steps-1) + visu_file, tree, container->size());
    }
#ifdef USE_CATALYST
    if(catalyst_enable)
    {
        catalyst_adaptor::execute(n_time_steps-1, tree);
        catalyst_adaptor::finalize();
    }
#endif

    // ---------------------------------------
    // write output fma
    if(!output_file.empty())
    {
        std::cout << "Write outputs in " << output_file << std::endl;
        scalfmm::io::FFmaGenericWriter<double> writer(output_file);
        writer.writeDataFromTree(tree, container->size());
    }
    // ---------------------------------------
    // write output fma
    if(!visu_file.empty())
    {
        std::cout << "Write VTK file in " << visu_file << std::endl;
        scalfmm::tools::io::exportVTKxml(visu_file, tree, container->size());
    }

    delete container;

    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, local_args::tree_height{}, local_args::order{},
                                           local_args::input_file{}, local_args::output_file{},
                                           local_args::block_size{}, local_args::dimension{}, local_args::time_steps{},
                                           local_args::visu_file{}, local_args::delta{}, local_args::catalyst{}, local_args::catalyst_script{});
    parser.parse(argc, argv);
    // Getting command line parameters

    const std::size_t dimension = parser.get<local_args::dimension>();

    switch(dimension)
    {
    case 2:
    {
        constexpr std::size_t dim = 2;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::others::grad_one_over_r2<dim>;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using far_matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using interpolator_type = scalfmm::interpolation::interpolator<double, dim, far_matrix_kernel_type
            , scalfmm::options::uniform_<scalfmm::options::fft_>>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;

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
