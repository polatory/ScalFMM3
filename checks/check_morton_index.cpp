#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "scalfmm/utils/sort.hpp"
#include "scalfmm/utils/tensor.hpp"
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

using Value_type = double;
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::output_file(),
                                                    args::tree_height{}, args::order{},   // args::thread_count{},
                                                    args::block_size{}, args::log_file{}, args::log_level{});
    parser.parse(argc, argv);
    // Getting command line parameters
    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';

    const int group_size{parser.get<args::block_size>()};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';

    //    const std::string input_file{parser.get<args::input_file>()};
    //    if(!input_file.empty())
    //    {
    //        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
    //        <<
    //        '\n';
    //    }

    const auto output_file{parser.get<args::output_file>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    // Parameter handling

    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    static constexpr std::size_t dimension = 2;
    static constexpr std::size_t inputs = 1;
    static constexpr std::size_t outputs = 1;
    const auto order{parser.get<args::order>()};
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    // Open particle file
    std::size_t number_of_particles{};
    cpp_tools::timers::timer time{};

    // ---------------------------------------
    // scalfmm 3.0 tree tests and benchmarks.
    // ---------------------------------------
    using particle_type = scalfmm::container::particle<Value_type, dimension, Value_type, inputs, Value_type, outputs>;
    //    using particle_type = scalfmm::container::particle<Value_type, dimension, Value_type,
    //    number_of_physical_values, Value_type, 1>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<scalfmm::component::grid_storage<double, dimension, inputs, outputs>>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "Creating & Inserting " << number_of_particles
              << "particles for version .0 ...\n"
              << cpp_tools::colors::reset;

    scalfmm::container::point<Value_type, dimension> box_center{};

    Value_type box_width{1.};
    // generate particles: one par leaf, the octree is full.

    Value_type step{box_width / std::pow(2, (tree_height - 1))};
    std::cout << "Step = " << step << '\n';

    auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
    std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';

    xt::xarray<std::tuple<Value_type, Value_type, Value_type, Value_type>> particles(
      std::vector(dimension, number_of_values_per_dimension));
    number_of_particles = particles.size();
    //    std::cout << "linspace = "
    //              << xt::linspace(Value_type(-box_width / 2.) + step * 0.5,
    //              Value_type(box_width / 2.) - step * 0.5,
    //                              number_of_values_per_dimension)
    //              << '\n';

    auto particle_generator = scalfmm::tensor::generate_meshgrid<dimension>(
      xt::linspace(Value_type(-box_width / 2.) + step * 0.5, Value_type(box_width / 2.) - step * 0.5,
                   number_of_values_per_dimension));
    auto eval_generator = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::eval(std::forward<decltype(xs)>(xs))...); }, particle_generator);
    auto flatten_views = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::flatten(std::forward<decltype(xs)>(xs))...); }, eval_generator);

    auto particle_flatten_views = xt::flatten(particles);

    container_type container(particles.size());
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        container.insert_particle(
          i, std::apply([&i](auto&&... xs) { return std::make_tuple(std::forward<decltype(xs)>(xs)[i]..., 0., 0.); },
                        flatten_views));
        std::cout << i << " p " << container.particle(i) << std::endl;
    }

    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    box_type box(box_width, box_center);
    std::array<bool, dimension> pbc;
    pbc.fill(false);
    group_tree_type gtree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                          static_cast<std::size_t>(group_size), container);
    ///
    ///
    /// Set in nodes the particles sorted according to their Morton index.
    ///
    auto leaf_level = gtree.leaf_level();
    // bool check_different_morton_index = false;
    // bool check_sort_container = false;
    bool check_morton_list = true;
    bool check_morton_grid = true;
    bool check_morton_neighbors = true;
    bool check_morton_m2l_list = true;
    // bool check = true;
    if(check_morton_list)
    {
        for(std::size_t i = 0; i < number_of_particles; ++i)
        {
            position_type nodes = container.position(i);
            auto host = scalfmm::index::get_coordinate_from_position_and_corner(
              nodes, box.corner(0), box.width(0),
              tree_height);   // m_tree_height ?? and not m_tree_height -1
            auto index = scalfmm::index::get_morton_index(nodes, box, leaf_level);
            std::cout << " nodes " << nodes << " morton : " << scalfmm::index::get_morton_index(nodes, box, leaf_level)
                      << " from host " << scalfmm::index::get_morton_index(host) << "  only pos: " << host
                      << " Coordinate: "
                      << scalfmm::index::get_coordinate_from_morton_index<dimension>(index)
                      //          << "  M2L " << std::endl;
                      << "  P2P " << std::endl;
            auto neig =
              //  scalfmm::index::get_interaction_neighbors(scalfmm::operators::impl::tag_m2l(), host, tree_height - 1);
              scalfmm::index::get_m2l_list(host, tree_height - 1, pbc, 1);
            const auto& indexes = std::get<0>(neig);
            auto idx_neig = std::get<2>(neig);
            std::cout << "   indexes            ";
            for(auto a: indexes)
            {
                std::cout << a << " ";
            }
            std::cout << std::endl;
            std::cout << "   indexes_in_array   ";
            std::cout << "   idx_neig           " << idx_neig << std::endl;
            //            std::cout << "  --------- new get_m2l_list ----------" << std::endl;
            //            auto neig1 = scalfmm::index::get_m2l_list(host, gtree.leaf_level());
            //            const auto& indexes1 = std::get<0>(neig1);
            //            const auto& indexes_in_array1 = std::get<1>(neig1);
            //            auto idx_neig1 = std::get<2>(neig1);
            //            if(idx_neig1 != idx_neig)
            //            {
            //                std::cout << "wrong number of elements" << std::endl;
            //                std::exit(EXIT_FAILURE);
            //            }
            //            bool check = true;
            //            for(int i = 0; i < idx_neig; ++i)
            //            {
            //                check = check && (indexes1[i] == indexes[i]);
            //            }
            //            if(check)
            //            {
            //                std::cout << "Same morton index" << std::endl;
            //            }
            //            else
            //            {
            //                std::cout << "Wrong morton index" << std::endl;
            //                std::exit(EXIT_FAILURE);
            //            }
            //            check = true;
            //            for(int i = 0; i < idx_neig; ++i)
            //            {
            //                check = check && (indexes_in_array1[i] == indexes_in_array1[i]);
            //            }
            //            if(check)
            //            {
            //                std::cout << "Same indexes_in_array1 index" << std::endl;
            //                        }
            //            else
            //            {
            //                std::cout << "Wrong indexes_in_array1 index" << std::endl;
            //                std::exit(EXIT_FAILURE);
            //            }
            //            std::cout << "   indexes1            ";
            //            for(auto a: indexes1)
            //            {
            //                std::cout << a << " ";
            //            }
            //            std::cout << std::endl;
            //            std::cout << "   indexes_in_array1   ";
            //            for(auto a: indexes_in_array1)
            //            {
            //                std::cout << a << " ";
            //            }
            //            std::cout << std::endl;
            //            std::cout << "   idx_neig1          " << idx_neig1 << std::endl;
            //            std::cout << std::endl;
        }
    }
    if(check_morton_grid)
    {
        using CoordinateType = int;
        std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
        for(CoordinateType idx = 0; idx < std::pow(2, dimension * (tree_height - 1)); ++idx)
        {
            auto ijk = scalfmm::index::get_coordinate_from_morton_index<dimension>(idx);
            // std::array<CoordinateType, dimension> ijk{};
            std::cout << "morton " << idx << " [";
            for(auto ii: ijk)
            {
                std::cout << ii << " ";
            }
            std::cout << "]" << std::endl;
        }
        if(check_morton_neighbors)
        {
            int level = 3;
            const int neighbour_separation = 1;
            auto getN = [&level, &pbc](std::size_t indexRef)
            {
                auto pos = scalfmm::index::get_coordinate_from_morton_index<2>(indexRef);

                auto neig1 = scalfmm::index::get_neighbors(pos, level, pbc, neighbour_separation);
                auto& indexes = std::get<0>(neig1);
                const auto& nb = std::get<1>(neig1);
                std::cout << "Neighbors of  " << indexRef << " are ";
                for(std::size_t i = 0; i < nb; ++i)
                {
                    std::cout << " " << indexes[i];
                }
                std::cout << '\n';
                std::sort(std::begin(indexes), std::begin(indexes) + nb);
                std::cout << "sorted Neighbors of  " << indexRef << " are (" << nb << ")";
                for(std::size_t i = 0; i < nb; ++i)
                {
                    std::cout << " " << indexes[i];
                }
                std::cout << '\n';
            };
            getN(0);
            getN(26);
            getN(51);
            getN(52);
            getN(60);
            level = 2;
            getN(13);
        }
        if(check_morton_m2l_list)
        {
            std::cout << "\n\n ------------- M2L list -------------\n";
            int level = 3;
            auto getNM2LI = [&level, &pbc](std::size_t indexRef)
            {
                std::cout << "-------------\n";
                auto pos = scalfmm::index::get_coordinate_from_morton_index<2>(indexRef);

                auto neig1 = scalfmm::index::get_m2l_list(pos, level, pbc, 1);
                auto& indexes = std::get<0>(neig1);
                // const auto& indexes_in_array1 = std::get<1>(neig1);
                const auto& nb = std::get<2>(neig1);
                std::cout << "List of  " << indexRef << " are (" << nb << ")";
                for(std::size_t i = 0; i < nb; ++i)
                {
                    std::cout << " " << indexes[i];
                }
                std::cout << '\n';
                //            std::sort(std::begin(indexes), std::begin(indexes) + nb);
                //            std::cout << "sorted Neighbors of  " << indexRef << " are ";
                //            for(std::size_t i = 0; i < nb; ++i)
                //            {
                //              std::cout << " " << indexes[i];
                //            }
                std::cout << '\n';
            };
            getNM2LI(0);
            getNM2LI(26);
            getNM2LI(51);
            getNM2LI(52);
            getNM2LI(59);
            getNM2LI(60);
            getNM2LI(63);
        }
    }
    return 0;
}
