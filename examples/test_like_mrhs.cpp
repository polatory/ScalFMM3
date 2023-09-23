#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
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

#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>

// #include <array>
// #include <chrono>
// #include <cstdio>
#include <iostream>
// #include <sstream>
#include <string>
// #include <sys/types.h>
// #include <thread>
// #include <tuple>
// #include <unistd.h>
// #include <utility>
// #include <vector>
/**
 * \brief Store the PerfTest program parameters.
 */
auto cli = cpp_tools::cl_parser::make_parser(args::tree_height{}, args::order{}, args::thread_count{},
                                             args::input_file{}, args::output_file{}, args::block_size{},
                                             args::log_file{}, args::log_level{}, cpp_tools::cl_parser::help{});

struct command_line_parameters
{
    explicit command_line_parameters(const decltype(cli)& cli)
    {
        tree_height = cli.get<args::tree_height>();
        order = cli.get<args::order>();
        thread_count = cli.get<args::thread_count>();
        input_file = cli.get<args::input_file>();
        output_file = cli.get<args::output_file>();
        log_file = cli.get<args::log_file>();
        log_level = cli.get<args::log_level>();
        block_size = cli.get<args::block_size>();
    }

    int tree_height = 5;            ///< Tree height.
    std::size_t order = 3;          ///< Tree height.
    int thread_count = 1;           ///< Maximum thread count (when used).
    std::string input_file = "";    ///< Particles file.
    std::string output_file = "";   ///< Output particule file.
    std::string log_file = "";      ///< Log file.
    std::string log_level = "";     ///< Log file.
    int block_size = 250;           ///< Group tree group size
};                                  // namespace args
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    cli.parse(argc, argv);
    command_line_parameters params(cli);

    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    // Getting command line parameters
    const auto tree_height{params.tree_height};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset << '\n';

    const auto group_size{params.block_size};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';

    const auto input_file{params.input_file};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset << '\n';
    }

    const auto output_file{params.output_file};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }

    static constexpr std::size_t dimension = 3;
    static constexpr std::size_t inputs = 2;
    static constexpr std::size_t outputs = 2;
    const auto runtime_order{params.order};
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << runtime_order << cpp_tools::colors::reset
              << '\n';

    // Open particle file
    std::size_t number_of_particles{};
    cpp_tools::timers::timer time{};

    // ---------------------------------------
    // scalfmm 3.0 tree tests and benchmarks.
    // ---------------------------------------
    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::like_mrhs;
    using interpolator_type = scalfmm::interpolation::interpolator<double, dimension, matrix_kernel_type, scalfmm::options::uniform_<scalfmm::options::fft_>>;
    using particle_type = scalfmm::container::particle<double, dimension, double, inputs, double, outputs, std::size_t>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;
    std::cout << cpp_tools::colors::green << "Creating & Inserting " << number_of_particles
              << "particles for version .0 ...\n"
              << cpp_tools::colors::reset;

    scalfmm::container::point<double, dimension> box_center{};

    double box_width{};
    bool verbose = true;

    scalfmm::io::FFmaGenericLoader<double, dimension> loader(input_file, verbose);

    box_center = scalfmm::container::point<double, dimension>{loader.getBoxCenter()[0], loader.getBoxCenter()[1],
                                                              loader.getBoxCenter()[2]};
    number_of_particles = loader.getNumberOfParticles();
    box_width = loader.getBoxWidth();
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << cpp_tools::colors::reset << '\n';
    time.tic();
    auto nb_val_to_red_per_part = loader.get_dimension() + loader.get_number_of_input_per_record();
    double* values_to_read = new double[nb_val_to_red_per_part]{};
    container_type container(number_of_particles);
    for(std::size_t idx = 0; idx < number_of_particles; ++idx)
    {
        loader.fillParticle(values_to_read, nb_val_to_red_per_part);
        particle_type p(position_type(0.), 0., 0., idx);
        std::size_t ii{0};
        for(auto& e: p.position())
        {
            e = values_to_read[ii++];
        }
        for(auto& e: p.inputs())
        {
            e = values_to_read[ii++];
        }
        container.insert_particle(idx, p);
    }
    time.tac();

    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    time.tic();
    box_type box(box_width, box_center);
    group_tree_type gtree(static_cast<std::size_t>(tree_height), runtime_order, box,
                          static_cast<std::size_t>(group_size), static_cast<std::size_t>(group_size), container);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Group tree created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    time.tic();
    using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
    using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
    //
    interpolator_type uniform(matrix_kernel_type{}, runtime_order, static_cast<std::size_t>(tree_height), box.width(0));
    near_field_type near_field(matrix_kernel_type{});
    far_field_type far_field(uniform);
    scalfmm::operators::fmm_operators<near_field_type, far_field_type> fmm_operator(near_field, uniform);

    time.tac();
    std::cout << cpp_tools::colors::yellow << "Kernel and Interp created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    auto operator_to_proceed = scalfmm::algorithms::all;
    scalfmm::algorithms::sequential::sequential(gtree, std::move(fmm_operator), operator_to_proceed);

    {
        std::cout << std::endl << " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& " << std::endl;
        std::cout << std::scientific;
        std::cout.precision(15);

        // Compute the Energy
        typename particle_type::outputs_type energy{};
        for(auto& e: energy)
        {
            e = 0.;
        };
        typename particle_type::inputs_type total_physical_value{};
        for(auto& e: total_physical_value)
        {
            e = 0.;
        };

        scalfmm::component::for_each_leaf(
          std::begin(gtree), std::end(gtree),
          [&energy, &total_physical_value](auto const& leaf)
          {
              auto res{laplace::compute_energy_tuple(matrix_kernel_type{}, leaf)};

              scalfmm::meta::repeat([](auto& e, auto o) { e += o; }, energy, std::get<0>(res));
              scalfmm::meta::repeat([](auto& e, auto o) { e += o; }, total_physical_value, std::get<1>(res));
          });
        scalfmm::meta::repeat(
          [](auto e) {
              std::cout << cpp_tools::colors::red << std::setprecision(10) << "Energy: " << e << cpp_tools::colors::reset
                        << '\n';
          },
          energy);
        scalfmm::meta::repeat(
          [](auto e) {
              std::cout << cpp_tools::colors::blue << std::setprecision(10) << "Total Physical Value: " << e
                        << cpp_tools::colors::reset << '\n';
          },
          total_physical_value);
    }

    return 0;
}
