// @FUSE_MPI
#include <array>
// #include <chrono>
// #include <thread>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/tools/fma_dist_loader.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/tree_io.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/dist_group_tree.hpp"
#include "scalfmm/tree/group_let.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/parameters.hpp"

#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/parallel_manager/parallel_manager.hpp>
///
/// \brief main
/// \param argv
/// \return
///
/// \code
///  ./check/RELEASE/test_build_let  --input-file ../data/debug/prolate.fma  --output-file res.fma
///       --order 3 --tree-height 4 --group-size 3 -d 3

///   mpirun -output-filename log --oversubscribe -np 3 ./check/RELEASE/test_build_let
///   --input-file ../buildMPI/prolate.fma --order 3 --tree-height 3
///   --group-size 3 -d 3
///   mpirun -output-filename log --oversubscribe -np 3 ./examples/Release/test_mpi_algo --input-file
///   ../data/debug/circle2d_r3.fma  --order 3 --tree-height 3   --group-size 3 -d 2
/// \endcode
namespace local_args
{
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
    // struct PartLeafDistrib
    // {
    //     /// Unused type, mandatory per interface specification
    //     using type = bool;
    //     /// The parameter is a flag, it doesn't expect a following value
    //     enum
    //     {
    //         flagged
    //     };
    //     cpp_tools::cl_parser::str_vec flags = {"--dist-part-leaf"};
    //     std::string description = "Use two distribution one for the particle and one for the tree";
    // };
}   // namespace local_args
template<int dimension>
auto run(cpp_tools::parallel_manager::parallel_manager& para, const std::string& input_file,
         const std::string& output_file, const int tree_height, const int& part_group_size, const int& leaf_group_size,
         const int order, const int level_shared, bool use_leaf_distribution, bool use_particle_distribution) -> int
{
    constexpr int nb_inputs_near = 1;
    constexpr int nb_outputs_near = 1;
    using value_type = double;
    using mortonIndex_type = std::size_t;
    using globalIndex_type = std::size_t;

    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

    using interpolator_type =
      scalfmm::interpolation::interpolator<double, dimension, matrix_kernel_type,
                                           scalfmm::options::uniform_<scalfmm::options::low_rank_>>;

    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs_near, value_type, nb_outputs_near
                                   //, mortonIndex_type, globalIndex_type
                                   >;
    // using read_particle_type = scalfmm::container::particle<value_type, dimension, value_type, nb_inputs_near,
    //                                                         value_type, 0, mortonIndex_type, globalIndex_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::dist_group_tree<cell_type, leaf_type, box_type>;

    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///   Read the data in parallel
    ///
    ///   1) read constants of the problem in file;
    ///   2) each processor read N/P particles

    scalfmm::io::DistFmaGenericLoader<value_type, dimension> loader(input_file, para, para.io_master());
    //
    const std::size_t number_of_particles = loader.getNumberOfParticles();
    const int local_number_of_particles = loader.getMyNumberOfParticles();
    value_type width = loader.getBoxWidth();
    auto centre = loader.getBoxCenter();
    auto nb_val_to_red_per_part = loader.get_dimension() + loader.get_number_of_input_per_record();
    double* values_to_read = new double[nb_val_to_red_per_part]{};
    box_type box(width, centre);

    const int leaf_level = tree_height - 1;
    //
    // const int rank = para.get_process_id();
    //
    std::vector<particle_type> particles_set(local_number_of_particles);
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

        particles_set[idx] = p;
        // std::cout << p << std::endl;
    }
    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // check
    //    for(std::size_t idx = 0; idx < 10; ++idx)
    //    {
    //        std::cout << idx << " p " << container.particle(idx) << std::endl;
    //    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///    Set particles in the tree and construct the let
    ///  1) sort the particles according to their Morton index
    ///  2) construct the tree and then the let
    ///
    // separation criteria used to construct M2L and P2P ghosts
    int separation = 1;
    // Construct the LET
    auto letGroupTree = scalfmm::tree::let::buildLetTree<group_tree_type>(
      para, number_of_particles, particles_set, box, leaf_level, level_shared, part_group_size, leaf_group_size, order,
      separation, use_leaf_distribution, use_particle_distribution);
    std::cout << " End construction of letGroupTree" << std::endl << std::flush;

    if(para.io_master())
    {
        std::cout << cpp_tools::colors::blue << "Print tree distribution\n";
        letGroupTree.print_distrib(std::cout);
        std::cout << cpp_tools::colors::reset;
    }
    letGroupTree.print_morton_ghosts(std::cout);

    // //
    // // Build interaction lists
    // int const& separation_criterion = separation;   // fmm_operator.near_field().separation_criterion();
    // bool const mutual = false;                      // fmm_operator.near_field().mutual();
    // scalfmm::list::sequential::build_interaction_lists(letGroupTree, letGroupTree, separation_criterion, mutual);

    scalfmm::io::trace(std::cout, letGroupTree, 2);

///
///////////////////////////////////////////////////////////////////////////////////////////////////////
///
///////////////////////////////////////////////////////////////////////////////////////////////////////
///   Save the data
#ifdef SCALFMM_DEBUG_MPI
    {
        const int rank = para.get_process_id();
        std::string outName1("tree_rank_" + std::to_string(rank) + ".bin");
        std::string header1("LOCAL TREE ");
        scalfmm::tools::io::save(outName1, letGroupTree, header1);
        const int nbDataPerRecord = scalfmm::container::particle_traits<particle_type>::number_of_elements;
        const int inputs_size = scalfmm::container::particle_traits<particle_type>::inputs_size;
        const bool verbose_write = false;   // True only for the master
        std::string outName2("particles_rank_" + std::to_string(rank) + ".fma");

        scalfmm::io::FFmaGenericWriter<value_type> writer_seq(outName2, verbose_write);
        // Get the number of particles
        auto number_of_particles = letGroupTree.number_particles();
        std::clog << "rank[" + std::to_string(rank) + "] number_of_particles " << number_of_particles << std::endl;
        ///
        writer_seq.writeDataFromTree(letGroupTree, number_of_particles);
    }
#endif
    // const int nbDataPerRecord = scalfmm::container::particle_traits<particle_type>::number_of_elements;
    // const int inputs_size = scalfmm::container::particle_traits<particle_type>::inputs_size;

    // // static constexpr std::size_t nbDataPerRecord = particle_type::number_of_elements;
    // scalfmm::tools::DistFmaGenericWriter<value_type> writer(output_file, para);
    // /// Get the number of particles
    // std::cout << "number_of_particles " << number_of_particles << std::endl;
    // ///
    // writer.writeHeader(centre, width, number_of_particles, sizeof(value_type), nbDataPerRecord, dimension,
    // inputs_size);
    // ///
    // writer.writeFromTree(letGroupTree, number_of_particles);
    // ///
    // ///////////////////////////////////////////////////////////////////////////////////////////////////////
    if(para.io_master())
        std::cout << "Save Tree in parallel\n";
    // std::string outName("saveTree_" + std::to_string(rank) + ".bin");
    std::string outName("saveTreeLet.bin");
    std::string header("Uniform LOW RANK ");
    scalfmm::tools::io::save(para, outName, letGroupTree, header);
    std::cout << "end save Tree in parallel\n";

    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    cpp_tools::parallel_manager::parallel_manager para;
    para.init();
    std::cout << std::boolalpha << "para.io_master() " << para.io_master() << " get_process_id() "
              << para.get_process_id() << std::endl;
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, args::input_file(), args::output_file(), args::tree_height{}, args::order{},
      args::thread_count{}, args::block_size{}, args::Dimension{}, local_args::PartDistrib{},
      /*local_args::PartLeafDistrib{},*/ local_args::LevelShared{});
    parser.parse(argc, argv);
    // Getting command line parameters
    const int tree_height{parser.get<args::tree_height>()};
    const int level_shared{parser.get<local_args::LevelShared>()};
    const int group_size{parser.get<args::block_size>()};
    const std::string input_file{parser.get<args::input_file>()};

    const auto output_file{parser.get<args::output_file>()};
    const auto order{parser.get<args::order>()};
    const auto dimension{parser.get<args::Dimension>()};

    bool use_particle_distribution{parser.exists<local_args::PartDistrib>()};
    bool use_leaf_distribution{!use_particle_distribution};
    // if(parser.exists<local_args::PartLeafDistrib>())
    // {
    //     use_leaf_distribution = true;
    //     use_particle_distribution = true;
    // }

    if(para.io_master())
    {
        std::cout << cpp_tools::colors::blue << "<params> Tree height: " << tree_height << cpp_tools::colors::reset
                  << '\n';
        std::cout << cpp_tools::colors::blue << "<params> Group Size:  " << group_size << cpp_tools::colors::reset
                  << '\n';
        std::cout << cpp_tools::colors::blue << "<params> order:       " << order << cpp_tools::colors::reset << '\n';
        if(!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input file:  " << input_file << cpp_tools::colors::reset
                      << '\n';
        }
        std::cout << cpp_tools::colors::blue << "<params> Output file: " << output_file << cpp_tools::colors::reset
                  << '\n';
        std::cout << cpp_tools::colors::blue << "<params> Particle Distribution: " << std::boolalpha
                  << use_particle_distribution << cpp_tools::colors::reset << '\n';
        std::cout << cpp_tools::colors::blue << "<params> Leaf Distribution:     " << std::boolalpha
                  << use_leaf_distribution << cpp_tools::colors::reset << '\n';
    }
    switch(dimension)
    {
    case 2:
    {
        constexpr int dim = 2;

        run<dim>(para, input_file, output_file, tree_height, group_size, group_size, order, level_shared,
                 use_leaf_distribution, use_particle_distribution);
        break;
    }
    case 3:
    {
        constexpr int dim = 3;

        run<dim>(para, input_file, output_file, tree_height, group_size, group_size, order, level_shared,
                 use_leaf_distribution, use_particle_distribution);
        break;
    }
    default:
    {
        std::cerr << "Dimension should be only 2 or 3 !!\n";
    }
    }
#ifdef SCALFMM_USE_MPI
    std::cout << std::flush;
    para.get_communicator().barrier();
#endif
    para.end();
}
