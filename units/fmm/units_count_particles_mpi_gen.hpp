// @FUSE_MPI
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/mpi/proc_task.hpp"
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/operators/count_kernel/count_kernel.hpp"
//
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_dist_loader.hpp"
#include "scalfmm/tools/tree_io.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/dist_group_tree.hpp"
#include "scalfmm/tree/group_let.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/generate.hpp"

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/parallel_manager/parallel_manager.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

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
//auto run(cpp_tools::parallel_manager::parallel_manager& para, const int& tree_height, const int& group_size,
//         Array const& pbc, const int nb_level_above_root, const bool readFile, std::string& input_file,
//         const bool interaction, bool use_leaf_distribution, bool use_particle_distribution) -> int
auto run(const int& tree_height, const int& group_size,  std::string& input_file,  bool use_leaf_distribution, bool mutual = false) -> int{
  static constexpr std::size_t number_of_physical_values = 1;
    static constexpr std::size_t dimpow2 = pow(2, Dimension);
    const auto runtime_order = 1;

    int level_shared{2};
    if(use_leaf_distribution) {
      use_particle_distribution = false ;
    }
    else {
      use_particle_distribution = true ;
    }
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
    }
    // std::cout << "pbc:    " << std::boolalpha;
    // for(auto e: pbc)
    // {
    //     std::cout << e << " ";
    // }
    // std::cout << std::endl;
    box_type box(box_width, box_center);
#ifdef SCALFMM_BUILD_PBC
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
    auto letTree = scalfmm::tree::let::buildLetTree<group_tree_type>(
      para, number_of_particles, particles_set, box, leaf_level, level_shared, group_size, group_size, runtime_order,
      separation, use_leaf_distribution, use_particle_distribution);

    //    if(para.io_master())
    {
        std::cout << cpp_tools::colors::blue << "Print tree distribution\n";
        letTree.print_distrib(std::cout);
        std::cout << cpp_tools::colors::reset;
    }

#ifdef SCALFMM_BUILD_PBC
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
    std::cout << cpp_tools::colors::red << "build_interaction_lists \n" << cpp_tools::colors::reset << std::flush;

    scalfmm::list::sequential::build_interaction_lists(letTree, letTree, separation_criterion, mutual);
    std::cout << cpp_tools::colors::red << "trace \n" << cpp_tools::colors::reset << std::flush;
    // if(para.io_master())
    {
        std::cout << cpp_tools::colors::red << "trace  2\n" << cpp_tools::colors::reset << std::flush;

        scalfmm::io::trace(std::cout, letTree, 2);
        std::cout << cpp_tools::colors::red << "trace  4\n" << cpp_tools::colors::reset << std::flush;

        scalfmm::io::trace(std::cout, letTree, 4);
    }

    //auto operator_to_proceed = scalfmm::algorithms::operators_to_proceed::farfield;
    // auto operator_to_proceed = scalfmm::algorithms::operators_to_proceed::nearfield;
    auto operator_to_proceed = scalfmm::algorithms::operators_to_proceed::all;
    //	 auto operator_to_proceed = (scalfmm::algorithms::operators_to_proceed::p2m | scalfmm::algorithms::operators_to_proceed::m2l);
    //	 auto operator_to_proceed = (scalfmm::algorithms::operators_to_proceed::p2m | scalfmm::algorithms::operators_to_proceed::m2m  | scalfmm::algorithms::operators_to_proceed::m2l)  ;

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
    if(right_number_of_particles)
    {
        std::cout << "Found the right number of particles - nb particles " << number_of_particles << std::endl;
    }
    else
    {
        std::cout << "wrong number of particles - nb particles (min) " << nb_particles_min << "  (max) "
                  << nb_particles_max << " (expected) " << number_of_particles << std::endl;

        if(para.io_master())
            std::cout << "Save Tree in parallel\n";
        // std::string outName("saveTree_" + std::to_string(rank) + ".bin");
        std::string outName("saveTreeLet.bin");
        std::string header("CHEBYSHEV LOW RANK ");
        scalfmm::tools::io::save(para, outName, letTree, header);
    }
    std::cout << cpp_tools::colors::reset << '\n';
    REQUIRE(right_number_of_particles);

    return right_number_of_particles;
}

//


//auto run(const int& tree_height, const int& group_size,  std::string& input_file,  bool use_leaf_distribution, bool mutual = false) -> int{
/*
TEMPLATE_TEST_CASE("test count 1d", "[test-count-1d]", double)
{
    SECTION("count 1d", "[count1d]") { run<1>(4, 10,  "../data/units/test_2d_ref.fma", false); }   // h = 5
}
*/
TEMPLATE_TEST_CASE("test count 2d", "[test-count-2d]", double)
{
  SECTION("count 2d", "[count2d]") { run<2>(3, 10, "../data/units/test_2d_ref.fma", false); }   // h = 5
}

TEMPLATE_TEST_CASE("test count 3d", "[test-count-3d]", double)
{
  // leaf distrubution and mutual
  SECTION("count 3d", "[count3d]") { run<3>(5, 40, "../data/units/sphere-706_source.fma", true,true); }   // h = 5
}

int main(int argc, char* argv[])
{
    int result = Catch::Session().run(argc, argv);

    return result;
}


