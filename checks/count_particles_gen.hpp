#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/operators/count_kernel/count_kernel.hpp"
//
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/parameters.hpp"
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
/// count_particles_{omp,seq} --input-file ../data/unitCube_100_PF.fma  --tree-height  3 -gs 2 --dimension 3
///  \endcode
///
///  * Here we generate one particle per leaf located at the center,
///      the number of particles = number of leaf = std::pow(2, dimension*(tree_height - 1))
/// \code
/// count_particles_{omp,seq} --tree-height  3 -gs 2 --dimension 2
///  \endcode

///
/// \brief The read_file struct
namespace local_args
{
    struct read_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description =
          "Input filename (.fma or .bfma). if not specified we construct one particles per leaves";
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

template<int Dimension, typename Array>
auto run(const int& tree_height, const int& group_size, Array const& pbc, const int nb_level_above_root,
         const bool readFile, std::string& input_file, const bool interaction) -> int
{
    static constexpr std::size_t number_of_physical_values = 1;
    const auto runtime_order = 1;
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
    using box3_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box3_type>;
    //
    // ------------------------------------------------------------------------------
    //
    scalfmm::container::point<double, Dimension> box_center(0.0);
    double box_width{1.};
    //
    container_type* container;
    std::size_t number_of_particles{};
    if(readFile)   // Read particles from a file
    {
        bool verbose = true;

        scalfmm::io::FFmaGenericLoader<double, Dimension> loader(input_file, verbose);

        number_of_particles = loader.getNumberOfParticles();
        box_width = loader.getBoxWidth();
        box_center = loader.getBoxCenter();

        auto nb_val_to_red_per_part = loader.get_dimension() + loader.get_number_of_input_per_record();
        double* values_to_read = new double[nb_val_to_red_per_part]{};
        container = new container_type(number_of_particles);

        for(std::size_t idx = 0; idx < number_of_particles; ++idx)
        {
            loader.fillParticle(values_to_read, nb_val_to_red_per_part);
            particle_type p;
            std::size_t ii{0};
            for(auto& e: p.position())
            {
                e = values_to_read[ii++];
            }
            container->insert_particle(idx, p);
        }
    }
    else
    {
        // generate particles: one par leaf, the octree is full.

        double step{box_width / std::pow(2, (tree_height - 1))};
        std::cout << "Step = " << step << '\n';

        auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
        std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';

        xt::xarray<std::tuple<double, double, double, double>> particles(
          std::vector(Dimension, number_of_values_per_dimension));
        number_of_particles = particles.size();
        //        std::cout << "linspace = "
        //                  << xt::linspace(double(-box_width / 2.) + step * 0.5, double(box_width / 2.) - step * 0.5,
        //                                  number_of_values_per_dimension)
        //                  << '\n';

        auto particle_generator = scalfmm::tensor::generate_meshgrid<Dimension>(xt::linspace(
          double(-box_width / 2.) + step * 0.5, double(box_width / 2.) - step * 0.5, number_of_values_per_dimension));
        auto eval_generator =
          std::apply([](auto&&... xs) { return std::make_tuple(xt::eval(std::forward<decltype(xs)>(xs))...); },
                     particle_generator);
        auto flatten_views = std::apply(
          [](auto&&... xs) { return std::make_tuple(xt::flatten(std::forward<decltype(xs)>(xs))...); }, eval_generator);

        auto particle_flatten_views = xt::flatten(particles);
        container = new container_type(particles.size());
        auto container_it = std::begin(*container);
        for(std::size_t i = 0; i < particles.size(); ++i)
        {
            *container_it =
              std::apply([&i](auto&&... xs) { return std::make_tuple(std::forward<decltype(xs)>(xs)[i]..., 0., 0.); },
                         flatten_views);
            //      std::cout << particle_type(*container_it) << '\n';
            ++container_it;
        }
    }
    std::cout << "pbc:    " << std::boolalpha;
    for(auto e: pbc)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;
    box3_type box3(box_width, box_center);
#ifdef scalfmm_BUILD_PBC
    box3.set_periodicity(pbc);
#endif
    std::cout << "Box: " << box3 << std::endl;
    group_tree_type tree(static_cast<std::size_t>(tree_height), runtime_order, box3,
                         static_cast<std::size_t>(group_size), static_cast<std::size_t>(group_size), *container);
#ifdef scalfmm_BUILD_PBC
    tree.set_levels_above_root(nb_level_above_root);
#endif
    //
    ///////////////////////////////////
    // using fmm_operator_type = count_kernels::particles::count_fmm_operator<Dimension>;
    // // fmm_operator_type fmm_operator{};
    using fmm_operator_type = scalfmm::operators::fmm_operators<count_kernels::particles::count_near_field,
                                                                count_kernels::particles::count_far_field<Dimension>>;
    bool mutual = false;

    count_kernels::particles::count_near_field nf(mutual);
    count_kernels::particles::count_far_field<Dimension> ff{};
    fmm_operator_type fmm_operator(nf, ff);
    auto operator_to_proceed = scalfmm::algorithms::all;
    // auto neighbour_separation = fmm_operator.near_field().separation_criterion();
    // tree.build_interaction_lists(tree, neighbour_separation, fmm_operator.near_field().mutual(),
    //                             scalfmm::list::policies::sequential);
#ifdef COUNT_USE_OPENMP
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](tree, fmm_operator, operator_to_proceed);
#else
    scalfmm::algorithms::sequential::sequential(tree, fmm_operator, operator_to_proceed);
#endif
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
    scalfmm::component::for_each_leaf(
      std::begin(tree), std::end(tree),
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

    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, local_args::dimension(), local_args::read_file(),
                                        /*local_args::check_interactions(),*/ args::tree_height{}, args::block_size{},
#ifdef COUNT_USE_OPENMP
                                        args::thread_count{},
#endif
                                        args::log_file{}, args::log_level{},
#ifdef scalfmm_BUILD_PBC
                                        args::pbc{},
#endif
                                        args::extended_tree_height{}   // periodicity
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
    std::cout << "pbc: " << std::boolalpha;
    for(auto e: pbc)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;
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

    if(dimension == 1)
    {
        run<1>(tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction);
    }
    else if(dimension == 2)
    {
        run<2>(tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction);
    }
    else if(dimension == 3)
    {
        run<3>(tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction);
    }
    else if(dimension == 4)
    {
        run<4>(tree_height, group_size, pbc, nb_level_above_root, readFile, input_file, interaction);
    }
}
