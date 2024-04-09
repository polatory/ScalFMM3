// @FUSE_MPI
#include <algorithm>
#include <vector>

//
// ScalFMM includes
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/tools/fma_dist_loader.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/group_let.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/parameters.hpp"
// for out::print
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/parallel_manager/parallel_manager.hpp>
//
// Maphyspp includes
//
//#include <maphys.hpp>
//#include <maphys/loc_data/DenseMatrix.hpp>

///
/////////////////////////////////////////////////////////////////
///
/// Example to run the program
///
/// seq
///   ./examples/Release/test-compose  --input-file ../data/debug/circle2d_r3.fma  --dimension 2 --tree-height 2
///
/// parallel
///
/////////////////////////////////////////////////////////////////

namespace local_args
{
    struct dimension
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n   2 for dimension 2, 3 for dimension 3";
        using type = int;
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
}   // namespace local_args

template<std::size_t Dimension, typename ContainerType, typename ValueType>
std::size_t read_data(cpp_tools::parallel_manager::parallel_manager& para, const std::string& filename, ContainerType& container,
                      scalfmm::container::point<ValueType, Dimension>& Centre, ValueType& width)
{
    using particle_type = typename ContainerType::value_type;

    scalfmm::io::DistFmaGenericLoader<ValueType, Dimension> loader(filename, para, para.io_master());

    const std::int64_t number_of_particles = loader.getNumberOfParticles();
    std::int64_t local_number_of_particles = loader.getMyNumberOfParticles();

    width = loader.getBoxWidth();
    Centre = loader.getBoxCenter();
    auto nb_val_to_red_per_part = loader.getNbRecordPerline();

    double* values_to_read = new double[nb_val_to_red_per_part]{0};
    container.resize(local_number_of_particles);
    std::cout << "number_of_particles " << number_of_particles << std::endl;
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
            e = ValueType(1.0);
        }
        container[idx] = p;
    }
    return number_of_particles;
}
template<typename ContainerType, typename Box_type, typename Distrib_type>
void build_distrib(cpp_tools::parallel_manager::parallel_manager& para, ContainerType& particle_container, std::size_t& number_of_particles,
                   const Box_type& box, const int& level, Distrib_type& index_blocs)
{
    auto rank = para.get_process_id();

    //  using morton_type = std::size_t;
    using index_type = std::int64_t;
    using morton_distrib_type = std::array<index_type, 2>;

    ///
    ///  sort particles according to their Morton index
    scalfmm::tree::let::sort_particles_inplace(para, particle_container, box, level);
    ///
    const std::size_t localNumberOfParticles = particle_container.size();
    std::vector<index_type> leafMortonIdx(localNumberOfParticles);
    // As the particles are sorted the leafMortonIdx is sorted too
#pragma omp parallel for shared(localNumberOfParticles, box, level)
    for(std::size_t part = 0; part < particle_container.size(); ++part)
    {
        leafMortonIdx[part] = scalfmm::index::get_morton_index(particle_container[part].position(), box, level);
    }
    /// construct the particle distribution  particles_distrib and in
    ///  leafMortonIdx whe have the morton index  of the leaf = blocs
    std::vector<morton_distrib_type> particles_distrib(para.get_num_processes());
#ifdef LEAVES_DIST
    particles_distrib = std::move(scalfmm::tree::distrib::balanced_leaves(para, leafMortonIdx));
#else
    particles_distrib = std::move(
      scalfmm::tree::distrib::balanced_particles(para, particle_container, leafMortonIdx, number_of_particles));
    // the morton index of the leaf
    auto last = std::unique(leafMortonIdx.begin(), leafMortonIdx.end());
    leafMortonIdx.erase(last, leafMortonIdx.end());
#endif
    /// Put the particles on the good processor
    scalfmm::tree::distrib::fit_particles_in_distrib(para, particle_container, leafMortonIdx, particles_distrib, box,
                                                     level, number_of_particles);
    ///
    //    //#pragma omp parallel for shared(localNumberOfParticles, box, level)
    //    for(std::size_t part = 0; part < particle_container.size(); ++part)
    //    {
    //        std::cout << " new (" << rank << ")" << part << " " << particle_container[part]
    //                  << " m= " << scalfmm::index::get_morton_index(particle_container[part].position(), box, level)
    //                  << std::endl;
    //    }
    ///
    /// build the distribution of rows/column or columns per leaf or bloc )
    ///
    auto nbBlocs = leafMortonIdx.size();
    index_blocs.resize(nbBlocs + 1, 0);
    std::size_t row{0};
    index_blocs[nbBlocs] = particle_container.size();

    auto myDistrib = particles_distrib[rank];
    //    scalfmm::out::print("rank(" + std::to_string(rank) + ") particles_distrib: ", particles_distrib);
    //    scalfmm::out::print("rank(" + std::to_string(rank) + ") leafMortonIdx: ", leafMortonIdx);
    auto nb_part = particles_distrib.size();
    for(std::size_t b = 0; b < nbBlocs - 1; ++b)
    {
        while(scalfmm::index::get_morton_index(particle_container[row].position(), box, level) == leafMortonIdx[b])
        {
            row++;
        }
        index_blocs[b + 1] = row;
    }
    //    std::cout << "Bloc distribution of rows: " << nbBlocs << std::endl;
    //    scalfmm::out::print("rank(" + std::to_string(rank) + ") blocs: ", blocs);

    //    // Check the output
    //    std::cout << "Particles \n";
    //    for(std::size_t p = 0; p < particle_container.size(); ++p)
    //    {
    //        std::cout << p << " " << particle_container[p] << "  "
    //                  << scalfmm::index::get_morton_index(particle_container[p].position(), box, level) << std::endl;
    //    }
    //    //
    //    std::cout << "Bloc distribution of rows:\n ";
    //    scalfmm::out::print("rank(" + std::to_string(rank) + ") blocs: ", blocs);
}
template<typename ContainerType, typename MatrixKernelType>

void build_save_matrix(const std::string& output_file, ContainerType container, MatrixKernelType& mk)
{
    std::ofstream file(output_file);
    std::cout << "writing matrix in file " << output_file << "\n";
    file.setf(std::ios::fixed);
    file.precision(14);
    auto NNZ = container.size() * (container.size() - 1) / 2;
    file << "%%MatrixMarket matrix coordinate real symmetric\n%\n";
    file << container.size() << "  " << container.size() << "  " << NNZ << std::endl;
    for(std::size_t i = 1; i < container.size(); ++i)
    {
        for(std::size_t j = 0; j < i; ++j)
        {
            auto val = mk.evaluate(container[i].position(), container[j].position()).at(0);
            file << i << "  " << j << "  " << val << std::endl;
        }
    }
    file.close();
}
template<std::size_t Dimension, typename... Parameters>
auto run(cpp_tools::cl_parser::parser<Parameters...> const& parser, cpp_tools::parallel_manager::parallel_manager& para) -> int
{
    ////////////////////////////////////////////////
    ///
    ///
    using value_type = double;
    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    // ---------------------------------------
    //  The matrix kernel
    static constexpr std::size_t nb_inputs_near{matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{matrix_kernel_type::kn};
    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, std::size_t>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using box_type = scalfmm::component::box<position_type>;

    ///
    ///////////////////////////////////////////////////////////////////////////////////
    // parameters
    const std::string input_file{parser.template get<local_args::input_file>()};
    const std::string output_file{parser.template get<local_args::output_file>()};
    const auto tree_height{parser.template get<args::tree_height>()};

    /// Read the particles data
    scalfmm::container::point<value_type, Dimension> box_center{};
    value_type box_width{};
    std::vector<particle_type> container{};
    std::size_t number_of_particles{0};
    if(input_file.empty())
    {
        std::cerr << cpp_tools::colors::red << "input file is empty !\n";
        return 0;
    }
    else
    {
        number_of_particles = read_data(para, input_file, container, box_center, box_width);
    }
    box_type box(box_width, box_center);
    if(para.io_master())
    {
        std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
        std::cout << cpp_tools::colors::green << "Box center = " << box_center << " box width = " << box_width
                  << cpp_tools::colors::reset << '\n';
    }
    ////
    /// build the distribution of cells per processor
    ///  index_dist = first particle index inside the cell owened by the processor
    ///   particles inside cell k start at index_dist[k]
    ///   and its number is index_dist[k+1]  - index_dist[k]
    ///   Thanks to this we are able to construct my local bloc matrix
    ///
    auto level = tree_height - 1;
    std::vector<int> index_dist;
    build_distrib(para, container, number_of_particles, box, level, index_dist);
    //
    auto rank = para.get_process_id();

    scalfmm::out::print("rank(" + std::to_string(rank) + ") blocs distrib: ", index_dist);

    /// Construct the bloc matrix

    matrix_kernel_type mk;
    if(para.get_num_processes() == 1)
    {
        build_save_matrix(output_file, container, mk);
    }
    ////
    std::cout << " End run \n";

    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    cpp_tools::parallel_manager::parallel_manager para;
    para.init();
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, local_args::input_file{}, local_args::output_file{},
                                           local_args::dimension{}, args::tree_height{});
    parser.parse(argc, argv);
    const std::size_t dimension = parser.get<local_args::dimension>();

    switch(dimension)
    {
    case 2:
    {
        constexpr std::size_t dim = 2;

        return run<dim>(parser, para);
        std::cerr << " out run \n";
        break;
    }
    case 3:
    {
        constexpr std::size_t dim = 3;
        return run<dim>(parser, para);
        break;
    }
    default:
        std::cout << "check  1/r Kernel for dimension 2 and 3. Value is \n"
                  << "          2 for dimension 2 "
                  << "          3 for dimension 3 " << std::endl;
        break;
    }
    para.end();
}
