#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"

#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/vtk_writer.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/sort.hpp"

#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
// @FUSE_FFTW

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

    struct block_size : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--group-size", "-gs"};
        std::string description = "Group tree chunk size.";
        using type = std::size_t;
        type def = 250;
    };

    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n   2 for dimension 2, 3 for dimension 3";
        using type = std::size_t;
        type def = 1;
    };

    struct pbc : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--per", "--pbc"};                       /*!< The flags */
        std::string description = "The periodicity in each direction (0 no periocity)"; /*!< The description */
        std::string input_hint = "0,1,1";                                               /*!< The description */
        using type = std::vector<bool>;
    };

    struct visu_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--visu-file", "-vf"};
        std::string description = "Output VTK file.";
        using type = std::string;
    };

    auto cli = cpp_tools::cl_parser::make_parser(tree_height{}, order{}, input_file{}, output_file{}, block_size{},
                                                 dimension{}, pbc{}, visu_file{}, cpp_tools::cl_parser::help{});
}   // namespace local_args

template<typename Container>
auto read_data(const std::string& filename)
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using value_type = typename particle_type::position_value_type;
    static constexpr std::size_t dimension{particle_type::dimension};
    const bool verbose{true};

    scalfmm::io::FFmaGenericLoader<value_type, dimension> loader(filename, verbose);

    const std::size_t number_of_particles{loader.getNumberOfParticles()};
    std::cout << cpp_tools::colors::yellow << "[file][n_particles] : " << number_of_particles
              << cpp_tools::colors::reset << '\n';
    const auto width{loader.getBoxWidth()};
    std::cout << cpp_tools::colors::yellow << "[file][box_width] : " << width << cpp_tools::colors::reset << '\n';
    const auto center{loader.getBoxCenter()};
    std::cout << cpp_tools::colors::yellow << "[file][box_centre] : " << center << cpp_tools::colors::reset << '\n';

    auto nb_val_to_red_per_part = loader.getNbRecordPerline();
    // could be a problem for binary file (float double)
    std::vector<value_type> values_to_read(nb_val_to_red_per_part);

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
        for(auto& e: p.inputs())
        {
            e = values_to_read[ii++];
        }
        p.variables(values_to_read[ii++], idx, 1);
        container.insert_particle(idx, p);
    }
    return std::make_tuple(container, center, width);
}

template<typename Container, typename ValueType>
auto generate_replicated_dist(Container const& origin, ValueType width, std::vector<bool> pbc) -> Container
{
    using container_type = Container;
    using particle_type = typename Container::value_type;
    using position_type = typename particle_type::position_type;
    static constexpr std::size_t dimension{particle_type::dimension};

    // number of axes replicated
    std::size_t ones = std::count(std::begin(pbc), std::end(pbc), true);
    // number of times the distribution will be replicated : 3^(axes concerned)
    auto n_replicates = scalfmm::math::pow(3, ones);

    std::cout << cpp_tools::colors::cyan << "[periodicity][replicating origin] : " << n_replicates << " times."
              << cpp_tools::colors::reset << '\n';

    // new container with replicates
    container_type replicated_dist(n_replicates * origin.size());

    // vector storing all translation vectors.
    std::vector<position_type> us{};

    // loop indices for nd loops : [-1,2[
    std::array<int, dimension> starts{};
    std::array<int, dimension> stops{};
    starts.fill(-1);
    stops.fill(2);
    // convert pbc vector to int
    std::array<int, dimension> pbc_integer{};
    std::transform(std::begin(pbc), std::end(pbc), std::begin(pbc_integer), [](auto e) { return int(e); });

    // lambda to generate all translating vectors
    auto finding_us = [&us, &pbc_integer, width](auto... is)
    {
        // building unitary translation vector
        position_type u{ValueType(is)...};
        // building ans of unitary translation vector
        std::array<int, dimension> u_int{std::abs(is)...};
        // mask
        std::array<int, dimension> masked{};

        // applying logical between u_int and pbc_integer
        // this will result in a mask
        std::transform(std::begin(u_int), std::end(u_int), std::begin(pbc_integer), std::begin(masked),
                       [](auto a, auto b) { return a || b; });
        // if this mask is equal to the pbc vector, it means that the current
        // translation vector needs to be selected
        // note that we also select the vector that does not move the original box
        bool eq = std::equal(std::begin(pbc_integer), std::end(pbc_integer), std::begin(masked));
        if(eq)
        {
            // push the unitary translation vector scaled to box width
            us.push_back(u * width);
            std::cout << cpp_tools::colors::cyan << "[periodicity][applying u] : " << u * width
                      << cpp_tools::colors::reset << '\n';
        }
    };

    // call the finding_us lambda in a nd loop nest.
    scalfmm::meta::looper_range<dimension>{}(finding_us, starts, stops);

    // check if we have the good number of translation vector
    if(us.size() != n_replicates)
    {
        std::cerr << cpp_tools::colors::red << "[error] : number of translating vectors != number of replications."
                  << cpp_tools::colors::reset << '\n';
        std::exit(-1);
    }

    // lambda to replicate and move particles in new container
    // applying a translation vector
    std::size_t jump_replication{0};
    auto applying_us = [&replicated_dist, &origin, &jump_replication](auto u)
    {
        for(std::size_t i{0}; i < origin.size(); ++i)
        {
            auto p = origin.particle(i);
            // applying u
            p.position() = p.position() + u;
            // if u is different from the zeros vector we know it's not
            // the original box so set the flag to 0
            if(u != position_type(0.))
            {
                auto& is_origin = scalfmm::meta::get<2>(p.variables());
                is_origin = 0;
            };
            // insert new particle
            replicated_dist.insert_particle((jump_replication * origin.size()) + i, p);
        }
        // jump to next replicate
        ++jump_replication;
    };

    // for each translation vector we generate new particles
    std::for_each(std::begin(us), std::end(us), applying_us);
    // returning the new vector
    return replicated_dist;
}

template<std::size_t Dimension, typename ValueType, typename FmmOperator, typename... Parameters>
auto run(cpp_tools::cl_parser::parser<Parameters...> const& parser) -> void
{
    using value_type = ValueType;
    using near_matrix_kernel_type = typename FmmOperator::near_field_type::matrix_kernel_type;
    using far_field_type = typename FmmOperator::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;

    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;

    const auto tree_height{local_args::cli.get<local_args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "[params][tree_height] : " << tree_height << cpp_tools::colors::reset
              << '\n';
    const auto group_size{local_args::cli.get<local_args::block_size>()};
    std::cout << cpp_tools::colors::blue << "[params][group_size] : " << group_size << cpp_tools::colors::reset << '\n';
    const auto order{local_args::cli.get<local_args::order>()};
    std::cout << cpp_tools::colors::blue << "[params][order] : " << order << cpp_tools::colors::reset << '\n';
    const auto input_file{local_args::cli.get<local_args::input_file>()};
    std::cout << cpp_tools::colors::blue << "[params][input_file] : " << input_file << cpp_tools::colors::reset << '\n';
    const std::string visu_file(parser.template get<local_args::visu_file>());
    if(!visu_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "[params][visu_file] : " << visu_file << cpp_tools::colors::reset
                  << '\n';
    }

    const auto pbc(local_args::cli.get<local_args::pbc>());

    if(pbc.size() != Dimension)
    {
        std::cerr << cpp_tools::colors::red << "[error] wrong dimension for periodic vector!"
                  << cpp_tools::colors::reset << '\n';
        std::exit(-1);
    }
    std::cout << cpp_tools::colors::green << "[status] entering tree test...\n" << cpp_tools::colors::reset;

    //  The matrix kernel
    //
    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    std::cout << cpp_tools::colors::blue << "[matrix_kernel][inputs_near] : " << nb_inputs_near
              << cpp_tools::colors::reset << '\n';
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};
    std::cout << cpp_tools::colors::blue << "[matrix_kernel][outputs_near] : " << nb_outputs_near
              << cpp_tools::colors::reset << '\n';
    static constexpr std::size_t nb_inputs_far{far_matrix_kernel_type::km};
    std::cout << cpp_tools::colors::blue << "[matrix_kernel][inputs_far] : " << nb_inputs_far
              << cpp_tools::colors::reset << '\n';
    static constexpr std::size_t nb_outputs_far{far_matrix_kernel_type::kn};
    std::cout << cpp_tools::colors::blue << "[matrix_kernel][outputs_far] : " << nb_outputs_far
              << cpp_tools::colors::reset << '\n';

    cpp_tools::timers::timer time{};

    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, value_type, std::size_t, int>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "[status] creating & inserting particles...\n" << cpp_tools::colors::reset;

    scalfmm::container::point<value_type, Dimension> box_center{};
    value_type box_width{};
    container_type origin_dist{};

    //-----------------------------
    // reading data
    time.tic();
    std::tie(origin_dist, box_center, box_width) = read_data<container_type>(input_file);
    time.tac();
    std::cout << cpp_tools::colors::green << "[status] ...done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::magenta << "[timings][reading data] = ." << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    //-----------------------------
    // generate replicated distribution
    std::cout << cpp_tools::colors::green << "[status] creating replicated particles distribution...\n"
              << cpp_tools::colors::reset;
    time.tic();
    auto replicated_dist = generate_replicated_dist(origin_dist, box_width, pbc);
    time.tac();
    std::cout << cpp_tools::colors::green << "[status] ...done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::magenta << "[timings][constructing replicated distribution] = " << time.elapsed()
              << "ms\n"
              << cpp_tools::colors::reset;

    //-----------------------------
    // constructing original tree
    std::cout << cpp_tools::colors::green << "[status] creating original tree...\n" << cpp_tools::colors::reset;
    time.tic();
    box_type box_origin(box_width, box_center);
    group_tree_type tree_origin(tree_height, order, box_origin, group_size, group_size, origin_dist);
    time.tac();
    std::cout << cpp_tools::colors::green << "[status] ...done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::magenta << "[timings][original tree] = " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    //-----------------------------
    // constructing replicated tree
    std::cout << cpp_tools::colors::green << "[status] creating replicated tree...\n" << cpp_tools::colors::reset;
    time.tic();
    box_type box_replicated(box_width * 3, box_center);
    group_tree_type tree_replicated(tree_height, order, box_replicated, group_size, group_size, replicated_dist);
    time.tac();
    std::cout << cpp_tools::colors::green << "[status] ...done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::magenta << "[timings][replicated tree] = " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    //-----------------------------
    // export visu file
    if(!visu_file.empty())
    {
        scalfmm::tools::io::exportVTKxml("origin_" + visu_file, tree_origin, origin_dist.size());
        scalfmm::tools::io::exportVTKxml("replicated_" + visu_file, tree_replicated, replicated_dist.size());
    }
}

template<typename V, std::size_t D, typename MK>
using interpolator_alias =
  scalfmm::interpolation::interpolator<V, D, MK, scalfmm::options::uniform_<scalfmm::options::fft_>>;

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    using value_type = double;

    local_args::cli.parse(argc, argv);
    // Getting command line parameters

    using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
    using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;

    const auto dimension{local_args::cli.get<local_args::dimension>()};
    std::cout << cpp_tools::colors::blue << "[params][dimension] : " << dimension << cpp_tools::colors::reset << '\n';

    switch(dimension)
    {
    case 1:
    {
        static constexpr std::size_t dim{1};
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(local_args::cli);
        break;
    }
    case 2:
    {
        static constexpr std::size_t dim{2};
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(local_args::cli);
        break;
    }
    case 3:
    {
        static constexpr std::size_t dim{3};
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(local_args::cli);
        break;
    }
    case 4:
    {
        static constexpr std::size_t dim{4};
        //
        using interpolator_type = interpolator_alias<value_type, dim, matrix_kernel_type>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(local_args::cli);
        break;
    }
    default:
        std::cerr << cpp_tools::colors::red << "[error] wrong dimension !" << cpp_tools::colors::reset << '\n';
        return -1;
    }
    return 0;
}
