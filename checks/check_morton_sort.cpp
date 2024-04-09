
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "scalfmm/utils/sort.hpp"

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

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

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::input_file(), args::output_file(),
                                        args::tree_height{}, args::order{},   // args::thread_count{},
                                        args::block_size{}, args::log_file{}, args::log_level{}, isSorted{});
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
    // Parameter handling

    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    static constexpr std::size_t dimension = 3;
    static constexpr std::size_t inputs = 2;
    static constexpr std::size_t outputs = 2;
    const auto order{parser.get<args::order>()};
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    // Open particle file
    std::size_t number_of_particles{};
    cpp_tools::timers::timer time{};

    // ---------------------------------------
    // scalfmm 3.0 tree tests and benchmarks.
    // ---------------------------------------
    using particle_type = scalfmm::container::particle<double, dimension, double, inputs, double, outputs, std::size_t>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<scalfmm::component::grid_storage<double, dimension, inputs, outputs>>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "Creating & Inserting " << number_of_particles
              << "particles for version .0 ...\n"
              << cpp_tools::colors::reset;

    scalfmm::container::point<double, dimension> box_center{};

    double box_width{};
    bool verbose = true;

    scalfmm::io::FFmaGenericLoader<double> loader(input_file, verbose);

    box_center = scalfmm::container::point<double, dimension>{loader.getBoxCenter()[0], loader.getBoxCenter()[1],
                                                              loader.getBoxCenter()[2]};
    std::vector<bool> pbc(dimension, false);
    number_of_particles = loader.getNumberOfParticles();
    box_width = loader.getBoxWidth();
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << cpp_tools::colors::reset << '\n';
    time.tic();
    auto nb_val_to_red_per_part = loader.get_dimension() + loader.get_number_of_input_per_record();
    double* values_to_read = new double[nb_val_to_red_per_part]{};
    container_type* container = new container_type(number_of_particles);
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
        container->insert_particle(idx, p);
    }
    time.tac();

    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    box_type box(box_width, box_center);
    group_tree_type gtree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                          static_cast<std::size_t>(group_size), *container);
    ///
    ///
    /// Set in nodes the particles sorted according to their Morton index.
    ///
    auto leaf_level = gtree.leaf_level();
    bool check_different_morton_index = true;
    bool check_sort_container = false;
    bool check_morton_neighbors = true;

    if(check_sort_container)
    {
        auto perm = scalfmm::utils::get_morton_permutation(box, leaf_level, *container);
        std::vector<position_type> nodes(number_of_particles);
        container_type* ordered_container = new container_type(number_of_particles);

        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            const auto& p = perm[i];
            std::cout << i << " idx " << std::get<0>(p) << " morton " << std::get<1>(p) << std::endl;
            nodes[i] = container->position(std::get<0>(perm[i]));
            ordered_container->insert_particle(i, container->particle(std::get<0>(perm[i])));
        }
        container_type* old_container = container;

        container = ordered_container;
        ordered_container = nullptr;
        scalfmm::utils::check_positions(gtree, perm, nodes);
        std::cout << " New container \n";
        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            std::cout << " nodes " << nodes[i] << " morton "
                      << scalfmm::index::get_morton_index(nodes[i], box, leaf_level) << " particle "
                      << container->particle(i) << std::endl;
        }
        std::cout << " pointer on ordered_container " << old_container << std::endl;

        scalfmm::utils::sort_container(box, leaf_level, old_container);
        std::cout << "  Container after sort_container method \n";
        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            std::cout << " nodes " << nodes[i] << " morton "
                      << scalfmm::index::get_morton_index(nodes[i], box, leaf_level) << " particle "
                      << old_container->particle(i) << std::endl;
        }
        delete ordered_container;
        delete old_container;
    }
    if(check_different_morton_index)
    {
        const int neighbour_separation = 1; 
        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            position_type nodes = container->position(i);
            auto host = scalfmm::index::get_coordinate_from_position_and_corner(
              nodes, box.corner(0), box.width(0), tree_height);   // m_tree_height ?? and not m_tree_height -1
            auto index = scalfmm::index::get_morton_index(nodes, box, leaf_level);
            std::cout << " nodes " << nodes << " morton : " << scalfmm::index::get_morton_index(nodes, box, leaf_level)
                      << " from host " << scalfmm::index::get_morton_index(host) << "  only pos: " << host
                      << "Coordinate: " << scalfmm::index::get_coordinate_from_morton_index<dimension>(index)
                      << std::endl;
            auto neig = scalfmm::index::get_neighbors(host, tree_height, pbc, neighbour_separation);
            const auto& indexes = std::get<0>(neig);
            std::cout << "   indexes            ";
            for(auto a: indexes)
            {
                std::cout << a << " ";
            }
            std::cout << std::endl;
        }
    }
    if(check_morton_neighbors)
    {
        std::cout << "par 0 " << container->particle(0) << std::endl;
        auto index = scalfmm::index::get_morton_index(container->position(0), box, leaf_level);
        // Calculate the coordinate of the component from the morton index
        auto coordinate{scalfmm::index::get_coordinate_from_morton_index<dimension>(index)};
    }
    return 0;
}
