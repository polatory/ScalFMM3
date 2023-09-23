
#include <iostream>
#include <string>
#include <vector>

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/tikz_writer.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/io.hpp"
#include "scalfmm/tree/leaf_view.hpp"

#include "scalfmm/utils/sort.hpp"
#include "tools_param.hpp"

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

/// \file  trace_tree.cpp
//!
//! \brief trace_tree: Driver to print tree information and interactions lists (p2p and m2l).
//!
//! \code
//! USAGE:
//!  ./tools/Release/trace_tree [--help] --dim int --input-file value [--output-file value] --tree-height value [--per 0,1,1] \
//!         [--group-size value] [--not-mutual] [--group-interactions] [--log-level value] [--display-particles] [--color value]
//! DESCRIPTION:
//!     --help, -h
//!         Display this help message
//!     --dim, --dimension, -d int
//!         Dimension of the space (1,2 or 3)
//!     --input-file, -fin value
//!         Input particle filename (.fma or .bfma).
//!     --output-file, -fout value
//!         Generic file name without extention ( .txt the tree trace and in 2d .tex tikz plate of the leavs.
//!     --tree-height, -th value
//!         Tree height (or initial height in case of an adaptive tree).
//!     --per, --pbc 0,1,1
//!         The periodicity in each direction (0 no periodicity)
//!     --group-size, -gs value
//!         Group tree chunk size.
//!     --not-mutual
//!         Do not consider dependencies in p2p interaction list (not mutual
//!         algorithm)
//!     --log-level, -ll value
//!         Log level for trace
//!         1 and 2 tree information,
//!         3 + p2p interactions list
//!         4 + m2l interactions list
//!         5 p2p and m2l interactions list
//!     --display-particles, -dp
//!         Print the particles
//!     --color, -c value
//!         Color to print the morton index or the particles
//! \endcode
//! <b>examples</b>
//!  * To print the full p2p interaction list for a tree of height 4 in 2-d
//!  \code
//!   ./trace_tree --input-file  input_2d_file.fma --tree-height 4 --not-mutual --log-level 2 --dimension 2 
//!  \endcode
//!  the option --not-mutual removes the block to clock interactions
//!  * To print the  m2l block interaction list for a tree of height 4 in 2-d
//!  \code
//!   ./trace_tree --input-file  input_2d_file.fma --tree-height 4 --not-mutual --log-level 2 --dimension 2 
//!  \endcode 
//
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
        cpp_tools::cl_parser::str_vec flags = {"--data-sorted", "-ds"};
        std::string description = "Precise if the data are sorted by their morton index";
    };
    struct Notmutual
    {
        cpp_tools::cl_parser::str_vec flags = {"--not-mutual"};
        std::string description = "Do not consider dependencies in p2p interaction list (not mutual algorithm)";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
    // struct grpInter
    // {
    //     /// Unused type, mandatory per interface specification
    //     using type = bool;
    //     /// The parameter is a flag, it doesn't expect a following value
    //     enum
    //     {
    //         flagged
    //     };
    //     cpp_tools::cl_parser::str_vec flags = {"--group-interactions", "-gi"};
    //     std::string description = "Build and display group interactions";
    // };
    struct displayParticles
    {
        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--display-particles", "-dp"};
        std::string description = "Print the particles";
    };
    struct color
    {
        /// Unused type, mandatory per interface specification
        using type = std::string;
        /// The parameter is a flag, it doesn't expect a following value
        cpp_tools::cl_parser::str_vec flags = {"--color", "-c"};
        std::string description = "Color to print the morton index or the particles (2d only)";
        type def = "blue";
    };
    ///  level_trace = 1 print minimal information (height, order, group size)
    ///  level_trace = 2 print information of the tree (group interval and index inside)
    ///  level_trace = 3 print information of the tree (leaf interval and index inside and their p2p interaction
    ///  list)
    /// level_trace = 4 print information of the tree (cell interval and index inside and their m2l
    ///  interaction list)
    /// level_trace = 5 print information of the tree (leaf and cell interval and index inside
    ///  and their p2p and m2l interaction lists)
    struct logLevel
    {
        /// Unused type, mandatory per interface specification
        using type = int;
        /// The parameter is a flag, it doesn't expect a following value
        cpp_tools::cl_parser::str_vec flags = {"--log-level", "-ll"};
        std::string description = "Log level for trace \n  1 and 2 tree information,\n"
                                  "3 + p2p interactions list\n"
                                  "4 + m2l interactions list\n"
                                  "5 p2p and m2l interactions list\n";
        ;
        type def = 2;
    };
    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description =
          "Generic file name without extention ( .txt the tree trace and in 2d .tex tikz plate of the leaves ).";
        using type = std::string;
    };
}   // namespace local_args

template<int Dimension, typename Vector>
auto run_trace(const std::string& input_file, std::string const& output_file, const int tree_height,
               const int group_size, std::string const& color, bool displayParticles, Vector const& pbc, bool mutual,
               int log_level) -> int
{
    std::cout << cpp_tools::colors::blue << "Entering run_trace...\n" << cpp_tools::colors::reset;

    static constexpr std::size_t inputs = 1;
    static constexpr std::size_t outputs = 1;

    const auto order{2};
    // std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';
    //   const auto log_level{parser.get<args_tools::log_level>()};
    // Open particle file
    std::size_t number_of_particles{};

    // ---------------------------------------
    // scalfmm 3.0 tree tests and benchmarks.
    // ---------------------------------------
    using particle_type = scalfmm::container::particle<double, Dimension, double, inputs, double, outputs, std::size_t>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<scalfmm::component::grid_storage<double, Dimension, inputs, outputs>>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "Creating & Inserting " << number_of_particles
              << "particles for version .0 ...\n"
              << cpp_tools::colors::reset;

    // scalfmm::container::point<double, Dimension> box_center{};

    double box_width{};
    bool verbose = true;
    scalfmm::io::FFmaGenericLoader<double, Dimension> loader(input_file, verbose);
    if(loader.get_dimension() != Dimension)
    {
        throw std::invalid_argument(
          "The dimension is wrong (dimension in file different thant the dimension in argument!");
    }
    position_type box_center = loader.getBoxCenter();
    number_of_particles = loader.getNumberOfParticles();
    box_width = loader.getBoxWidth();
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << cpp_tools::colors::reset << '\n';

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
        if(nb_val_to_red_per_part > Dimension)
        {
            for(auto& e: p.inputs())
            {
                e = values_to_read[ii++];
            }
        }
        p.variables(idx);
        container->insert_particle(idx, p);
    }

    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;

    box_type box(box_width, box_center);
    std::cout << "box" << box << std::endl;
    //    std::cout << *container << std::endl;
    group_tree_type tree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                         static_cast<std::size_t>(group_size), *container);
    const int separation_criterion = 1;

    scalfmm::list::sequential::build_interaction_lists(tree, tree, separation_criterion, mutual);

    if(Dimension == 2 && !output_file.empty())
    {
    
        std::string tikzName(output_file + ".tex");
        scalfmm::tools::io::exportTIKZ(tikzName, tree, displayParticles, color, true);
    }

    if(output_file.empty())
    {
        scalfmm::io::trace(std::cout, tree, log_level);
    }
    else
    {
        std::cout << "Write interactions in " << output_file + ".txt" << std::endl;
        std::ofstream file(output_file + ".txt");
        scalfmm::io::trace(file, tree, log_level);
    }
    return 0;
}
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, args_tools::dimensionSpace(), args_tools::input_file_req(),
      local_args::output_file(), args_tools::tree_height{}, args_tools::pbc{}, args_tools::block_size{},
      local_args::Notmutual{}, /*local_args::grpInter{}, */local_args::logLevel{}, local_args::displayParticles{},
      local_args::color{});

    parser.parse(argc, argv);
    // Getting command line parameters
    int dimension{parser.get<args_tools::dimensionSpace>()};

    std::cout << cpp_tools::colors::blue << "<params> Dimension : " << dimension << cpp_tools::colors::reset << '\n';

    const int tree_height{parser.get<args_tools::tree_height>()};

    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';

    const int group_size{parser.get<args_tools::block_size>()};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';

    const std::string input_file{parser.get<args_tools::input_file_req>()};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                  << '\n';
    }

    const std::string output_file{parser.get<local_args::output_file>()};

    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }

    const std::string color(parser.get<local_args::color>());
    bool displayParticles{parser.exists<local_args::displayParticles>()};

    std::vector<bool> pbc(dimension, false);
    if(parser.exists<args_tools::pbc>())
    {
        pbc = parser.get<args_tools::pbc>();

        if(pbc.size() != 2)
        {
            std::cerr << "Only works in 2 d \n";
            exit(-1);
        }
    }
    const bool mutual{!parser.exists<local_args::Notmutual>()};
    std::cout << cpp_tools::colors::blue << "<params> Mutual:     " << std::boolalpha << mutual
              << cpp_tools::colors::reset << '\n';
    auto log_level{parser.get<local_args::logLevel>()};
    std::cout << cpp_tools::colors::blue << "<params> log_level:           " << log_level << cpp_tools::colors::reset
              << '\n';
    // Parameter handling
    if(dimension == 2)
    {
        return run_trace<2>(input_file, output_file, tree_height, group_size, color, displayParticles, pbc, mutual,
                            log_level);
    }
    else if(dimension == 3)
    {
        return run_trace<3>(input_file, output_file, tree_height, group_size, color, displayParticles, pbc, mutual,
                            log_level);
    }
    else if(dimension == 1)
    {
        return run_trace<1>(input_file, output_file, tree_height, group_size, color, displayParticles, pbc, mutual,
                            log_level);
    }
    else if(dimension == 4)
    {
        return run_trace<4>(input_file, output_file, tree_height, group_size, color, displayParticles, pbc, mutual,
                            log_level);
    }
    return 1;
}
