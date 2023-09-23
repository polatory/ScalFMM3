

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/count_kernel/count_kernel.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "scalfmm/utils/source_target.hpp"
#include "scalfmm/utils/tensor.hpp"

#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <initializer_list>

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
///         Log to file using spdlog.
///
///         --log-level, -llog value
/// \endcode
///
/// Examples
///  * we count the number of particles from the input file
/// \code
/// ./examples/RelWithDebInfo/count_kernel --input-file ../data/unitCubeXYZQ100_sorted.bfma  --tree-height  3 -gs 2
///  \endcode
///
///  * Here we generate one particle per leaf located at the center,
///      the number of particles = number of leaf = std::pow(2, dimension*(tree_height - 1))
/// \code
/// ./examples/RelWithDebInfo/count_kernel --tree-height  3 -gs 2
///  \endcode

///
/// \brief The read_file struct
namespace local_args
{

    struct input_source_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-source-file", "-isf"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };
    struct input_target_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-target-file", "-itf"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };
    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "-d"};
        std::string description = "Dimension : \n   0 for dimension 2, 1 for dimension 3";
        using type = int;
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
    std::cout << " nb_val_to_red_per_part " << nb_val_to_red_per_part << std::endl;
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
        if(nb_val_to_red_per_part > dimension)
        {
            for(auto& e: p.inputs())
            {
                e = values_to_read[ii++];
            }
        }
        else
        {
            for(auto& e: p.inputs())
            {
                e = 0;
            }
        }
        // p.variables(values_to_read[ii++], idx, 1);
        p.variables(idx);
        container[idx] = p;
    }
        return std::make_tuple(container, center, width);
}

template<typename Container, typename Value_type>
auto gen_data(Value_type const& box_width, int const& tree_height, Value_type const& decal)
{
    using particle_type = typename Container::value_type;
    static constexpr std::size_t dimension{particle_type::dimension};
    static constexpr std::size_t dimpow2 = pow(2, dimension);
    using point_type = scalfmm::container::point<Value_type, dimension>;

    // step is the leaf size
    Value_type step{box_width / std::pow(2, (tree_height - 1))};
    auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
    std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';
    std::cout << "Step = " << step << '\n';
    // start is the center of the first box
    auto start = -box_width * 0.5 + step * 0.5;
    auto delta = step * 0.25 * decal;   // used to separate source and target
    auto number_of_particles = std::pow(dimpow2, (tree_height - 1));
    std::cout << "number_of_particles = " << number_of_particles << " box_width " << box_width << '\n';

    Container particles(number_of_particles);
    for(std::size_t index{0}; index < number_of_particles; ++index)
    {
        // get coord of the cell in the grid with the morton index
        auto coord = scalfmm::index::get_coordinate_from_morton_index<dimension>(index);

        point_type pos{coord};
        //
        particle_type p;
        int ii{0};
        for(auto& e: p.position())
        {
            e = start + step * pos[ii++] + delta;
        }
        particles[index] = p;
    }
    return particles;
}
template<int Dimension, typename value_type /*, typename Array*/>
auto run(const int& tree_height, const int& group_size, /*Array const& pbc, const int nb_level_above_root,*/
         const bool readFile, std::string& input_source_file, std::string& input_target_file, const bool mutual)
  ->int
{
    const auto order = 1;
    // Parameter handling

    cpp_tools::timers::timer<std::chrono::minutes> time{};

    constexpr int zeros{1};
    static constexpr std::size_t nb_inputs{1};
    static constexpr std::size_t nb_outputs{1};
    static constexpr std::size_t number_of_physical_values = 1;

    ///////////////////////////////////////////////////////////////////////////
    using point_type = scalfmm::container::point<value_type, Dimension>;
    using particle_source_type =
      scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs, value_type, zeros, std::size_t>;
    using particle_target_type =
      scalfmm::container::particle<value_type, Dimension, value_type, zeros, value_type, nb_outputs, std::size_t>;
    ///////////////////////////////////////////////////////////////////////////

    // Construct the container of particles
    // using container_type = scalfmm::container::particle_container<particle_source_type>;
    using container_source_type = std::vector<particle_source_type>;
    using container_target_type = std::vector<particle_target_type>;
    using box_type = scalfmm::component::box<point_type>;
    //
    using leaf_source_type = scalfmm::component::leaf_view<particle_source_type>;
    using leaf_target_type = scalfmm::component::leaf_view<particle_target_type>;
    //

    using cell_type =
      scalfmm::component::cell<scalfmm::component::grid_storage<value_type, Dimension, number_of_physical_values, 1>>;

    using tree_source_type = scalfmm::component::group_tree_view<cell_type, leaf_source_type, box_type>;
    using tree_target_type = scalfmm::component::group_tree_view<cell_type, leaf_target_type, box_type>;
    //
    // ------------------------------------------------------------------------------
    //
    std::cout << cpp_tools::colors::green << "Creating & Inserting ...\n" << cpp_tools::colors::reset;
    //
    time.tic();
    point_type box_center_source{0.};
    value_type box_width_source{1.0};
    // container_type container_source{};
    container_source_type container_source;
    // std::tie(container_source, box_center_source, box_width_source) = read_data<container_type>(input_source_file);
    // box_type box_source(box_width_source, box_center_source);

    point_type box_center_target{0.};
    value_type box_width_target{1.0};
    // container_type container_target{};
    container_target_type container_target;

    if(readFile)
    {
        if(!input_source_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input sources file: " << input_source_file
                      << cpp_tools::colors::reset << '\n';
            throw std::invalid_argument("No input sources file");
        }
        std::tie(container_source, box_center_source, box_width_source) =
          read_data<container_source_type>(input_source_file);
        if(!input_target_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input targets file : " << input_target_file
                      << cpp_tools::colors::reset << '\n';
            throw std::invalid_argument("No input targets file");
        }
        std::tie(container_target, box_center_target, box_width_target) =
          read_data<container_target_type>(input_target_file);
    }
    else
    {
        box_width_target = value_type(1.0);
        box_width_source = value_type(1.0);
        container_target = gen_data<container_target_type>(box_width_target, tree_height, value_type(+1.0));
        container_source = gen_data<container_source_type>(box_width_source, tree_height, value_type(-1.0));
    }
    box_type box_source(box_width_source, box_center_source);
    box_type box_target(box_width_target, box_center_target);

    time.tac();
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    //
    std::size_t number_of_particles{container_source.size()};
    std::cout << "box_source " << box_source << std::endl;
    std::cout << "box_target " << box_target << std::endl;
    //
    auto box = scalfmm::utils::bounding_box(box_source, box_target);
    std::cout << "Bounding_box " << box << std::endl;
    // auto container_all = scalfmm::utils::merge(container_source, container_target);
    // std::cout << container_all << std::endl;

    // build trees
    bool sorted = false;
    tree_source_type tree_source(tree_height, order, box, group_size, group_size, container_source, sorted);

    tree_target_type tree_target(tree_height, order, box, group_size, group_size, container_target, sorted);

    //
    ///////////////////////////////////
    // using fmm_operator_type = count_kernels::particles::count_fmm_operator<Dimension>;
    // fmm_operator_type fmm_operator(mutual);
    using fmm_operator_type = scalfmm::operators::fmm_operators<count_kernels::particles::count_near_field,
                                                                count_kernels::particles::count_far_field<Dimension>>;
    count_kernels::particles::count_near_field nf(mutual);
    count_kernels::particles::count_far_field<Dimension> ff{};
    fmm_operator_type fmm_operator(nf, ff);

    auto operator_to_proceed = scalfmm::algorithms::all;
    // auto separation_criterion = fmm_operator.near_field().separation_criterion();

    // tree_target.build_interaction_lists(tree_source, separation_criterion, mutual);

#ifdef COUNT_USE_OPENMP
    scalfmm::algorithms::omp::task_dep(tree_source, tree_target, fmm_operator, operator_to_proceed);
#else
    scalfmm::algorithms::sequential::sequential(tree_source, tree_target, fmm_operator, operator_to_proceed);
#endif

    std::size_t nb_particles_min = 2 * number_of_particles, nb_particles_max = 0;
    bool right_number_of_particles = true;

    scalfmm::component::for_each_leaf(
      std::begin(tree_target), std::end(tree_target),
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
    if(right_number_of_particles)
    {
        std::cout << "Found the right number of particles - nb particles " << number_of_particles << std::endl;
    }
    else
    {
        std::cout << "wrong number of particles - nb particles (min) " << nb_particles_min << "  (max) "
                  << nb_particles_max << " (expected) " << number_of_particles << std::endl;
    }
    // std::string outputFile{"count_target.fma"};
    // std::cout << "Write targets in " << outputFile << std::endl;
    // scalfmm::io::FFmaGenericWriter<double> writer_t(outputFile);
    // writer_t.writeDataFromTree(tree_target, container_target.size());

    // outputFile = "count_source.fma";
    // scalfmm::io::FFmaGenericWriter<double> writer_s(outputFile);
    // writer_s.writeDataFromTree(tree_source, container_source.size());
    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, local_args::dimension(), local_args::input_source_file(),
      local_args::input_target_file(), args::tree_height{}, args::block_size{},
#ifdef COUNT_USE_OPENMP
      args::thread_count{},
#endif
      args::log_file{}, args::log_level{}   // #ifdef SCALFMM_BUILD_PBC
                                            //                                         ,args::pbc{},
                                            // args::extended_tree_height{}   // periodicity
                                            // #endif
    );

    parser.parse(argc, argv);

    // Getting command line parameters
    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';

    const int group_size{parser.get<args::block_size>()};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';
    int dimension = parser.get<local_args::dimension>();

#ifdef COUNT_USE_OPENMP
    const std::size_t nb_threads{parser.get<args::thread_count>()};
    omp_set_dynamic(0);
    omp_set_num_threads(nb_threads);
#endif
    const bool readFile{parser.exists<local_args::input_source_file>() and
                        parser.exists<local_args::input_source_file>()};
    std::string input_source_file, input_target_file;
    bool interaction = false;

    if(readFile)
    {
        input_source_file = parser.get<local_args::input_source_file>();
        input_target_file = parser.get<local_args::input_target_file>();
    }

    if(dimension == 1)
    {
        run<1, double>(tree_height, group_size, /*pbc, nb_level_above_root,*/ readFile, input_source_file,
                       input_target_file, interaction);
    }
    else if(dimension == 2)
    {
        run<2, double>(tree_height, group_size, /*pbc, nb_level_above_root,*/ readFile, input_source_file,
                       input_target_file, interaction);
    }
    else if(dimension == 3)
    {
        run<3, double>(tree_height, group_size, /*pbc, nb_level_above_root,*/ readFile, input_source_file,
                       input_target_file, interaction);
    }
}
