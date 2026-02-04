// @FUSE_MPI
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/mpi/proc_task.hpp"
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/operators/count_kernel/count_kernel.hpp"
//
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/meta/const_functions.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_dist_loader.hpp"
#include "scalfmm/tools/tree_io.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/dist_group_tree.hpp"
#include "scalfmm/tree/group_let.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/parameters.hpp"

#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/parallel_manager/parallel_manager.hpp>

///
/// usage
/// \code
/// count_kernel  --help
/// USAGE:
///    count_kernel [--help] [--input-file value] --tree-height value [--threads value] [--group-size value] [--log-file
///    value] [--log-level value] ///
///     DESCRIPTION:
///
///     --help, -h
///             Display this help message
///
///         --input-file, -fin value
///             Input filename (.fma or .bfma). if not specified we construct one
///         particles per leaves
///
///         --tree-height, -th value
///      Tree height (or initial height in case of an adaptive tree).
///
///      --threads, -t value
///         Maximum thread count to be used.
///
///         --group-size, -gs value
///         Group tree chunk size.
///
///         --log-file, -flog value
///         Log to file using log.
///
///         --log-level, -llog value
/// \endcode
///
/// Examples
///
///  * Here we generate one particle per leaf located at the center,
///      the number of particles = number of leaf = std::pow(2, dimension*(tree_height - 1))
/// \code
/// count_particles_mpi --tree-height  3 -gs 2 --dimension 2 --dist-part-leaf
///  \endcode

/**
 *  mpirun --bind-to none  --oversubscribe --output TAG -np 3 ./checks/Release/count_particles_mpi -d 2   --input-file ../data/debug/circle2d_r3.fma --tree-height 3 -gs 7 --dist-part -t 3
*/
///
/// \brief The read_file struct
namespace local_args
{
    struct read_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        const char* description =
          "Input filename (.fma or .bfma). if not specified we construct one particles per leaves";
        using type = std::string;
    };
    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "-d"};
        const char* description = "Dimension : \n   0 for dimension 2, 1 for dimension 3";
        using type = int;
    };
    struct LevelShared
    {
        cpp_tools::cl_parser::str_vec flags = {"--level_shared", "-ls"};
        std::string description = "For this and those above we consider a sequential distribution (none = -1)";
        using type = int;
        type def = 2;
    };
    struct PartDistrib
    {
        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--dist-part"};
        std::string description = "Use the particle distribution to distribute the tree";
    };
    struct PartLeafDistrib
    {
        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--dist-part-leaf"};
        std::string description = "Use two distribution one for the particle | one for the tree";
    };
}   // namespace local_args

template<std::size_t dim>
inline auto constexpr get_accumulate_shape()
{
    if constexpr(dim == 1)
    {
        return std::array<std::size_t, dim>{1};
    }
    if constexpr(dim == 2)
    {
        return std::array<std::size_t, dim>{1, 1};
    }
    if constexpr(dim == 3)
    {
        return std::array<std::size_t, dim>{1, 1, 1};
    }
    if constexpr(dim == 4)
    {
        return std::array<std::size_t, dim>{1, 1, 1, 1};
    }
}

template<int Dimension, typename Array>
auto run(cpp_tools::parallel_manager::parallel_manager& para, const int& tree_height, const int& group_size,
         Array const& pbc, const int nb_level_above_root, const bool readFile, std::string& input_file,
         const bool interaction, bool use_leaf_distribution, bool use_particle_distribution) -> int
{
    static constexpr std::size_t number_of_physical_values = 1;
    static constexpr std::size_t dimpow2{scalfmm::meta::pow(2, Dimension)};
    const auto runtime_order = 1;

    int level_shared{2};

    //
    const int rank = para.get_process_id();
    const int nproc = para.get_num_processes();
    // Parameter handling
    if(readFile)
    {
        if(!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                      << '\n';
        }
    }

    // ------------------------------------------------------------------------------
    using particle_type = scalfmm::container::particle<double, Dimension, double, number_of_physical_values, double, 1>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type =
      scalfmm::component::cell<scalfmm::component::grid_storage<double, Dimension, number_of_physical_values, 1>>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::dist_group_tree<cell_type, leaf_type, box_type>;
    //
    // ------------------------------------------------------------------------------
    //
    using point_type = scalfmm::container::point<double, Dimension>;
    point_type box_center(0.0);
    double box_width{1.};
    //
    container_type* container;
    std::vector<particle_type> particles_set;

    std::size_t number_of_particles{};
    std::size_t local_number_of_particles{};
    if(readFile)   // Read particles from a file
    {
        bool verbose = false;

        scalfmm::io::DistFmaGenericLoader<double, Dimension> loader(input_file, para, verbose);

        number_of_particles = loader.getNumberOfParticles();
        local_number_of_particles = loader.getMyNumberOfParticles();
        number_of_particles = loader.getNumberOfParticles();
        box_width = loader.getBoxWidth();
        box_center = loader.getBoxCenter();

        auto nb_val_to_red_per_part = loader.get_dimension() + loader.get_number_of_input_per_record();
        double* values_to_read = new double[nb_val_to_red_per_part]{};
        container = new container_type(local_number_of_particles);
        particles_set.resize(local_number_of_particles);
        for(std::size_t idx = 0; idx < local_number_of_particles; ++idx)
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
            // container->insert_particle(idx, p);
            particles_set[idx] = p;
        }
    }
    else
    {
        // generate particles: one par leaf, the octree is full.
        number_of_particles = std::pow(dimpow2, (tree_height - 1));
        std::cout << "number_of_particles = " << number_of_particles << " box_width " << box_width << '\n';

        auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
        const std::size_t bloc = number_of_particles / nproc;

        local_number_of_particles = (rank < nproc - 1) ? bloc : number_of_particles - (nproc - 1) * bloc;
        particles_set.resize(local_number_of_particles);

        //
        const std::size_t start_index{rank * bloc};
        const std::size_t end_index{start_index + local_number_of_particles};
        std::cout << "start_index = " << start_index << " end_index = " << end_index << '\n';

        double step{box_width / std::pow(2, (tree_height))};

        std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';
        std::cout << "Step = " << step << '\n';

        for(std::size_t index{start_index}, idx{0}; index < end_index; ++index, ++idx)
        {
            auto coord = scalfmm::index::get_coordinate_from_morton_index<Dimension>(index);

            point_type pos{coord};
            std::cout << idx << "index " << index << " coord " << coord << " centre: " << step * pos << std::endl;
            particle_type p;
            std::size_t ii{0};
            for(auto& e: p.position())
            {
                e = -box_width * 0.5 + step * 0.5 + step * pos[ii++];
            }
            particles_set[idx] = p;
        }
        // for(std::size_t i = 0; i < particles_set.size(); ++i)
        // {
        //     std::cout << i << " " << particles_set[i] << std::endl;
        // }
    }
    // std::cout << "pbc:    " << std::boolalpha;
    // for(auto e: pbc)
    // {
    //     std::cout << e << " ";
    // }
    // std::cout << std::endl;
    box_type box(box_width, box_center);
#ifdef scalfmm_BUILD_PBC
    box.set_periodicity(pbc);
#endif
    std::cout << "Box: " << box << std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///    Set particles in the tree and construct the let
    ///  1) sort the particles according to their Morton index
    ///  2) construct the tree, then the let
    ///
    const int leaf_level = tree_height - 1;
    // separation criteria used to construct M2L | P2P ghosts
    int separation = 1;
    // Construct the LET
    std::cout << "\n build let \n" << std::flush;

    auto letTree = scalfmm::tree::let::buildLetTree<group_tree_type>(
      para, number_of_particles, particles_set, box, leaf_level, level_shared, group_size, group_size, runtime_order,
      separation, use_leaf_distribution, use_particle_distribution);

    //    if(para.io_master())
    {
        std::cout << cpp_tools::colors::blue << "Print tree distribution\n" << std::flush;
        letTree.print_distrib(std::cout);
        std::cout << "\n trace  2\n" << std::flush;

        scalfmm::io::trace(std::cout, letTree, 2);
        std::cout << cpp_tools::colors::reset;
    }

#ifdef scalfmm_BUILD_PBC
    std::cerr << cpp_tools::color::red << "Doesn't work with PBC \n" << cpp_tools::color::reset;
    letTree.set_levels_above_root(nb_level_above_root);
#endif
    //
    ///////////////////////////////////
    // using fmm_operator_type = count_kernels::particles::count_fmm_operator<Dimension>;
    // // fmm_operator_type fmm_operator{};
    using fmm_operator_type = scalfmm::operators::fmm_operators<count_kernels::particles::count_near_field,
                                                                count_kernels::particles::count_far_field<Dimension>>;
    bool mutual = false;
    int const& separation_criterion = separation;   // fmm_operator.near_field().separation_criterion();

    count_kernels::particles::count_near_field nf(mutual);
    count_kernels::particles::count_far_field<Dimension> ff{};
    fmm_operator_type fmm_operator(nf, ff);
    std::cout << cpp_tools::colors::blue << "Fmm with kernels: " << nf.matrix_kernel().name() << " mutual "
              << std::boolalpha << cpp_tools::colors::reset;

    scalfmm::list::sequential::build_interaction_lists(letTree, letTree, separation_criterion, mutual);
    // std::cout << cpp_tools::colors::red << "trace \n" << cpp_tools::colors::reset << std::flush;
    // // if(para.io_master())
    //  {
    //      std::cout << cpp_tools::colors::red << "trace  4\n" << cpp_tools::colors::reset << std::flush;

    //      scalfmm::io::trace(std::cout, letTree, 4);
    //      std::cout << "\n &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& \n" << std::flush;

    //      std::cout << std::flush;
    //  }
    //| scalfmm::algorithms::operators_to_proceed::m2l
    //  auto operator_to_proceed =
    //    scalfmm::algorithms::operators_to_proceed::p2p | scalfmm::algorithms::operators_to_proceed::p2m |
    //    scalfmm::algorithms::operators_to_proceed::m2m | scalfmm::algorithms::operators_to_proceed::m2l |
    //    scalfmm::algorithms::operators_to_proceed::l2l | scalfmm::algorithms::operators_to_proceed::l2p;
    //  auto operator_to_proceed = scalfmm::algorithms::operators_to_proceed::farfield;
    //  auto operator_to_proceed = scalfmm::algorithms::operators_to_proceed::nearfield;
    auto operator_to_proceed = scalfmm::algorithms::operators_to_proceed::all;

    scalfmm::algorithms::mpi::proc_task(letTree, fmm_operator, operator_to_proceed);
    //
    std::size_t nb_particles_min = 20 * number_of_particles, nb_particles_max = 0, nb_per = 1;
    bool right_number_of_particles = true;
    int nb_part_above = 1;
    // Construct the total number of particles
    for(int d = 0; d < Dimension; ++d)
    {
        if(pbc[d])
        {
            nb_per *= 3;
            nb_part_above *= std::pow(2, nb_level_above_root + 1);
        }
    }
    number_of_particles *= nb_per;
    number_of_particles *= nb_part_above;
    scalfmm::component::for_each_mine_leaf(
      letTree.begin_mine_leaves(), letTree.end_mine_leaves(),
      [&right_number_of_particles, number_of_particles, &nb_particles_max, &nb_particles_min](auto const& leaf)
      {
          size_t nb_part = std::get<0>(*scalfmm::container::outputs_begin(leaf.particles()));
          nb_particles_max = std::max(nb_particles_max, nb_part);
          nb_particles_min = std::min(nb_particles_min, nb_part);
          if(nb_part != number_of_particles)
          {
              std::cout << cpp_tools::colors::red << "wrong number of particles - index " << leaf.index()
                        << " nb particles " << nb_part << std::endl;
              right_number_of_particles = false;
          }
      });
    std::cout << cpp_tools::colors::reset << '\n';

    // if(!right_number_of_particles)
    // {
    //     std::cout << "wrong number of particles - nb particles (min) " << nb_particles_min << "  (max) "
    //               << nb_particles_max << " (expected) " << number_of_particles << std::endl;

    //     //  if(para.io_master())
    //     //      std::cout << "Save Tree in parallel\n";
    //     //  // std::string outName("saveTree_" + std::to_string(rank) + ".bin");
    //     //  std::string outName("tree_count_mpi.bin");
    //     //  std::string header("CHEBYSHEV LOW RANK ");
    //     //  scalfmm::tools::io::save(para, outName, letTree, header);
    // }
    // else
    // {
    //     std::cout << "Found the right number of particles - nb particles " << number_of_particles << std::endl;
    // }

    // for(int level = letTree.height() - 1; level >= 2; --level)
    // {
    //     std::cout << "\n  --  level " << level << "   --  " << std::endl;
    //     scalfmm::component::for_each_mine_leaf(letTree.begin_mine_cells(level), letTree.end_mine_cells(level),
    //                                            [](auto const& cell)
    //                                            {
    //                                                std::cout << "cell index " << cell.index() << "  multipoles "
    //                                                          << cell.multipoles().at(0) << "  locals "
    //                                                          << cell.locals().at(0) << std::endl;
    //                                            });
    // }
    int res{int(right_number_of_particles)};

    para.get_communicator().reduce(&res, 1, MPI_INT, MPI_LAND, 0);

    if(para.io_master())
    {
        if(bool(res))
        {
            std::cout << cpp_tools::colors::blue << "Right number of particles  " << std::boolalpha << bool(res)
                      << cpp_tools::colors::reset << std::endl;
        }
        else
        {
            std::cout << cpp_tools::colors::red << "Wrong number of particles  " << std::boolalpha << bool(res)
                      << cpp_tools::colors::reset << std::endl;
        }
    }
    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    cpp_tools::parallel_manager::parallel_manager para;
    para.init();

    std::cout << "nproc: " << para.get_num_processes() << std::boolalpha << "para.io_master() " << para.io_master()
              << " get_process_id() " << para.get_process_id() << std::endl;

    // Parameter handling
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, local_args::dimension(), local_args::read_file(),
                                        args::tree_height{}, args::block_size{}, args::thread_count{},
#ifdef scalfmm_BUILD_PBC
                                        args::pbc{},
#endif
                                        args::extended_tree_height{},   // periodicity
                                        local_args::PartDistrib{}, local_args::PartLeafDistrib{});

    parser.parse(argc, argv);

    // Getting command line parameters
    const int tree_height{parser.get<args::tree_height>()};
    // const int level_shared{parser.get<local_args::LevelShared>()};
    const int group_size{parser.get<args::block_size>()};

    bool use_particle_distribution{parser.exists<local_args::PartDistrib>()};
    bool use_leaf_distribution{!use_particle_distribution};
    if(parser.exists<local_args::PartLeafDistrib>())
    {
        use_leaf_distribution = true;
        use_particle_distribution = false;
    }

    int dimension = parser.get<local_args::dimension>();
    //  OpenMP
    const std::size_t nb_threads{parser.get<args::thread_count>()};
    omp_set_dynamic(0);
    omp_set_num_threads(nb_threads);
    if(para.io_master())
    {
        std::cout << cpp_tools::colors::blue << "<params> Tree height: " << tree_height << cpp_tools::colors::reset
                  << '\n';

        std::cout << cpp_tools::colors::blue << "<params> Group Size:  " << group_size << cpp_tools::colors::reset
                  << '\n';
        std::cout << cpp_tools::colors::blue << "<params> Threads num: " << nb_threads << cpp_tools::colors::reset
                  << '\n';
    }
    //
    const bool readFile(parser.exists<local_args::read_file>());
    std::string input_file;
    bool interaction = false;
    // parser.exists<local_args::check_interactions>();
    std::vector<bool> pbc(dimension, false);
    int nb_level_above_root{-1};
#ifdef scalfmm_BUILD_PBC
    if(parser.exists<args::pbc>())
    {
        auto tmp = parser.get<args::pbc>();
        if(tmp.size() != dimension)
        {
            throw std::invalid_argument("The dimension of the pbc is wrong");
        }
        pbc = parser.get<args::pbc>();
        nb_level_above_root = parser.get<args::extended_tree_height>();
    }
    if(para.io_master())
    {
        std::cout << "pbc: " << std::boolalpha;
        for(auto e: pbc)
        {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
#endif
    if(readFile)
    {
        input_file = parser.get<local_args::read_file>();
        if(!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                      << '\n';
        }
    }
    //
    if(dimension == 1)
    {
        run<1>(para, tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction,
               use_leaf_distribution, use_particle_distribution);
    }
    else if(dimension == 2)
    {
        run<2>(para, tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction,
               use_leaf_distribution, use_particle_distribution);
    }

    else if(dimension == 3)
    {
        run<3>(para, tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction,
               use_leaf_distribution, use_particle_distribution);
    }
    else if(dimension == 4)
    {
        run<4>(para, tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction,
               use_leaf_distribution, use_particle_distribution);
    }

    //
    std::cout << std::flush;
    para.get_communicator().barrier();
    para.end();
}