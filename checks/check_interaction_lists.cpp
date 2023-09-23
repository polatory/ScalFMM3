// @FUSE_OMP
//
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include <scalfmm/container/iterator.hpp>
//
//
#include "scalfmm/meta/type_pack.hpp"
//
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
//
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/operators/p2p.hpp"
//
// Tree
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"
// Lists
#include "scalfmm/lists/policies.hpp"
#include "scalfmm/lists/omp.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/lists/utils.hpp"
//
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/laplace_tools.hpp"

#include "scalfmm/utils/compare_results.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/source_target.hpp"
//
#include "interactions_results.hpp"
// parameters
#include "scalfmm/utils/parameters.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

using namespace scalfmm::io;
/// \file  check_interaction_lists.cpp
//!
//! \brief check_interaction_lists on circle_100
//!
//! source != target check_interaction_lists -gs 5 --check-target
//! source == target && not mutual check_interaction_lists -gs 5
//! source == target &&  mutual check_interaction_lists -gs 5 -- mutual
//! \code
//! USAGE:
//! USAGE:
//!  ./examples/Release/check_interaction_lists [--help] [--group-size value] [--check-target] [--mutual]
//!
//! DESCRIPTION:
//!
//!     --help, -h
//!         Display this help message
//!
//!     --group-size, -gs value
//!         Group tree chunk size.
//!
//!     --check-target
//!         Check for source target
//!
//!     --mutual
//!         Don't consider dependencies in p2p (mutual algorithm)
namespace local_args
{
    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n  -  1 <dimension <4";
        using type = int;
        type def = 1;
    };
    struct mutual
    {
        cpp_tools::cl_parser::str_vec flags = {"--mutual"};
        std::string description = "Consider dependencies in p2p interaction list (mutual algorithm)";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
    struct check_target
    {
        cpp_tools::cl_parser::str_vec flags = {"--check-target"};
        std::string description = "Check for source target ";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
}   // namespace local_args

template<typename Tree>
auto print_leaves(Tree const& tree) -> void
{
    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&tree](auto& leaf) { scalfmm::io::print_leaf(leaf); });
}

template<typename TREE, typename ARRAY1, typename ARRAY2>
auto check_interactions(TREE& tree, ARRAY1 (*func_m2l)(std::size_t, int), ARRAY2 (*func_p2p)(std::size_t),
                        bool with_depend = false)
{
    bool res_p2p = true;
    bool res_m2l = true;
    auto const& top_level = tree.box().is_periodic() ? 1 : 2;
    //
    //  CHECK M2L INTERATION LISTS
    for(int level = tree.leaf_level(); level >= top_level; --level)
    {
        scalfmm::component::for_each_cell(
          tree.begin(), tree.end(), level,
          [&func_m2l, &res_m2l, &level](auto const& cell)
          {
              auto const& indexes = cell.csymbolics().interaction_indexes;
              auto interaction_morton_index = cell.csymbolics().interaction_iterators;
              // auto ref{st_get_symbolic_list_m2l(cell.index(), level)};
              auto ref{(func_m2l)(cell.index(), level)};
              for(std::size_t i{0}; i < cell.csymbolics().existing_neighbors; ++i)
              {
                  res_m2l = res_m2l && (ref.at(i) == indexes.at(i));
                  if(!res_m2l)
                  {
                      std::cout << "level " << level << " cell " << cell.index() << " nb elts "
                                << cell.csymbolics().existing_neighbors << "\n";
                      std::cout << "   ref: ";
                      scalfmm::io::print_array(ref, cell.csymbolics().existing_neighbors);
                      std::cout << "\n   comp: ";
                      scalfmm::io::print_array(indexes, cell.csymbolics().existing_neighbors);

                      std::cout << "\nWrong index from cell.csymbolics().interaction_indexes" << std::endl;
                      throw std::runtime_error("Wrong index from cell.csymbolics().interaction_indexes");
                  }
              }
              for(std::size_t i{0}; i < cell.csymbolics().existing_neighbors; ++i)
              {
                  res_m2l = res_m2l && (ref.at(i) == interaction_morton_index.at(i)->index());
                  if(!res_m2l)
                  {
                      std::cout << "level " << level << " cell " << cell.index() << "\n";
                      std::cout << "   ref: ";
                      scalfmm::io::print_array(ref, cell.csymbolics().existing_neighbors);
                      std::cout << "\n   comp: ";
                      for(std::size_t k{0}; k < cell.csymbolics().existing_neighbors; ++k)
                      {
                          std::cout << interaction_morton_index.at(k)->index() << " ";
                      }
                      std::cout << "\nWrong iterator index" << std::endl;
                      throw std::runtime_error("Wrong iterator index");
                  }
              }
          });
    }
    // Check P2P interaction list
    int count{0};
    scalfmm::component::for_each_leaf(
      tree.begin(), tree.end(),
      [&func_p2p, &res_p2p, &count, with_depend](auto const& leaf)
      {
          auto const& indexes = leaf.csymbolics().interaction_indexes;
          auto interaction_morton_index = leaf.csymbolics().interaction_iterators;
          // auto interaction_positions = leaf.csymbolics().interaction_positions;
          // auto ref{st_get_symbolic_list_p2p(leaf.index())};
          auto my_index = leaf.index();
          auto ref1{(func_p2p)(my_index)};
          std::vector<std::size_t> ref(ref1.size());
          int dim = 0;

          if(with_depend)
          {
              for(auto v: ref1)
              {
                  if(v < my_index)
                  {
                      ref[dim] = v;
                      ++dim;
                  }
                  else
                  {
                      break;
                  }
              }
              ref.resize(dim);
          }
          else
          {
              for(auto v: ref1)
              {
                  ref[dim] = v;
                  ++dim;
              }
          }

          if(leaf.csymbolics().existing_neighbors_in_group == 0)
          {
              std::clog << "Warning existing_neighbors_in_group is 0 in leaf " << leaf.index() << std::endl;
          };
          for(std::size_t i{0}; i < leaf.csymbolics().existing_neighbors_in_group; ++i)
          {
              res_p2p = res_p2p && (ref.at(i) == indexes.at(i));
              if(!res_p2p)
              {
                  std::cout << "leaf " << leaf.index() << "\n";
                  std::cout << "   ref: ";
                  print_array(ref, ref.size());
                  std::cout << "\n   comp: ";
                  print_array(indexes, leaf.csymbolics().existing_neighbors_in_group);

                  std::cout << "\nWrong index from leaf.csymbolics().interaction_indexes" << std::endl;
                  throw std::runtime_error("Wrong index from cell.csymbolics().interaction_indexes");
              }
          }
          for(std::size_t i{0}; i < leaf.csymbolics().existing_neighbors_in_group; ++i)
          {
              res_p2p = res_p2p && (ref.at(i) == interaction_morton_index.at(i)->index());
              if(!res_p2p)
              {
                  std::cout << "leaf " << leaf.index() << "\n";
                  std::cout << "   ref: ";
                  print_array(ref, leaf.csymbolics().existing_neighbors_in_group);
                  std::cout << "\n   comp: ";
                  for(std::size_t k{0}; k < leaf.csymbolics().existing_neighbors_in_group; ++k)
                  {
                      std::cout << interaction_morton_index.at(k)->index() << " ";
                  }
                  std::cout << "\nWrong iterator index" << std::endl;
                  //   if(count == 3)
                  throw std::runtime_error("Wrong iterator index");
              }
          }
          ++count;
      });
    return std::make_tuple(res_p2p, res_m2l);
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
        // p.variables(values_to_read[ii++], idx, 1);
        p.variables(idx);
        container.insert_particle(idx, p);
    }
    return std::make_tuple(container, center, width);
}

// check_source_target true when sources !:= targets
template<int dimension, typename value_type, class fmm_operators_type>
auto fmm_run(const std::string& input_source_file, const std::string& input_target_file, const int& tree_height,
             const int& group_size, const int& order, const bool check_source_target, const bool mutual) -> int
{
    bool display_container = false;
    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    //  The matrix kernel
    using near_matrix_kernel_type = typename fmm_operators_type::near_field_type::matrix_kernel_type;
    using far_field_type = typename fmm_operators_type::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;

    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
    static constexpr std::size_t nb_inputs{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs{far_matrix_kernel_type::kn};
    //
    // Open particles files

    cpp_tools::timers::timer<std::chrono::minutes> time{};

    constexpr int zeros{1};   // should be zero
    using point_type = scalfmm::container::point<value_type, dimension>;
    using particle_source_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, zeros, std::size_t>;
    using particle_target_type =
      scalfmm::container::particle<value_type, dimension, value_type, zeros, value_type, nb_outputs, std::size_t>;
    // Construct the container of particles
    using container_type = scalfmm::container::particle_container<particle_source_type>;
    using box_type = scalfmm::component::box<point_type>;
    //
    using leaf_source_type = scalfmm::component::leaf_view<particle_source_type>;
    using leaf_target_type = scalfmm::component::leaf_view<particle_target_type>;
    //

    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using tree_source_type = scalfmm::component::group_tree_view<cell_type, leaf_source_type, box_type>;
    using tree_target_type = scalfmm::component::group_tree_view<cell_type, leaf_target_type, box_type>;
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << cpp_tools::colors::green << "Creating & Inserting ...\n" << cpp_tools::colors::reset;
    //
    time.tic();
    point_type box_center_source{};
    value_type box_width_source{};
    container_type container_source{};
    std::tie(container_source, box_center_source, box_width_source) = read_data<container_type>(input_source_file);
    box_type box_source(box_width_source, box_center_source);

    point_type box_center_target{};
    value_type box_width_target{};
    container_type container_target{};
    std::tie(container_target, box_center_target, box_width_target) = read_data<container_type>(input_target_file);
    box_type box_target(box_width_target, box_center_target);
    time.tac();
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;

    if(display_container)
    {
        std::cout << "Read source particles \n";
        std::cout << "box_source " << box_source << std::endl;
        std::cout << container_source << std::endl;

        std::cout << cpp_tools::colors::green << "Box center = " << box_center_source
                  << " box width = " << box_width_source << cpp_tools::colors::reset << '\n';
        std::cout << "Read target particles \n";
        std::cout << "box_target " << box_target << std::endl;
        std::cout << container_target << std::endl;

        std::cout << cpp_tools::colors::green << "Box center = " << box_center_target
                  << " box width = " << box_width_target << cpp_tools::colors::reset << '\n';
    }
    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << " s\n"
              << cpp_tools::colors::reset;

    auto box = scalfmm::utils::bounding_box(box_source, box_target);
    std::cout << "bounding_box " << box << std::endl;
    // auto container_all = scalfmm::utils::merge(container_source, container_target) ;

    // build trees
    bool sorted = false;
    tree_source_type* tree;
    tree_source_type tree_source(tree_height, order, box, group_size, group_size, container_source, sorted);

    tree_target_type tree_target(tree_height, order, box, group_size, group_size, container_target, sorted);
    if(check_source_target)
    {
        tree = &tree_source;
    }
    else
    {
        tree = &tree_target;
    }
    const int separation_criterion = 1;
    // const bool mutual = false;
    //tree_target.build_interaction_lists(*tree, separation_criterion, mutual, scalfmm::list::policies::omp);
    // scalfmm::list::sequential::build_interaction_lists(*tree, tree_target,  separation_criterion, mutual);
    scalfmm::list::omp::build_interaction_lists(*tree, tree_target,  separation_criterion, mutual);

    if(mutual)
    {
        scalfmm::list::reconstruct_p2p_mutual_interaction_lists(tree_target);
    }
    //tree_target.trace(3);   // P2P
    //
    bool res_p2p{}, res_m2l{};
    if(check_source_target)
    {
        // source and target
        std::tie(res_p2p, res_m2l) =
          check_interactions(tree_target, &st_get_symbolic_list_m2l, &st_get_symbolic_list_p2p);
    }
    else
    {
        // Source == target
        std::tie(res_p2p, res_m2l) =
          check_interactions(tree_target, &t_get_symbolic_list_m2l, &t_get_symbolic_list_p2p, mutual);
    }
    if(res_p2p && res_m2l)
    {
        std::cout << "Interaction lists are valid!\n";
    }
    else
    {
        if(res_p2p)
        {
            std::cout << "P2P interaction list is VALID!\n";
        }
        else
        {
            std::cout << "P2P interaction list is WRONG!\n";
        }
        if(res_m2l)
        {
            std::cout << "M2L interaction list is VALID!\n";
        }
        else
        {
            std::cout << "M2L interaction list is WRONG!\n";
        }
    }
    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    using value_type = double;

    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{},
      args::block_size{},   // local_args::input_source_file(), local_args::input_target_file(),
      local_args::check_target{}, local_args::mutual{});

    parser.parse(argc, argv);
    std::string input_source_file{};
    std::string input_target_file{};

    const int tree_height{4};
    const int group_size{parser.get<args::block_size>()};
    const bool mutual{parser.exists<local_args::mutual>()};
    const bool check_source_target{parser.exists<local_args::check_target>()};

    const auto order{4};
    std::cout << cpp_tools::colors::blue << "<params> Tree height:         " << tree_height << cpp_tools::colors::reset
              << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Group Size:          " << group_size << cpp_tools::colors::reset
              << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Mutual:              " << std::boolalpha << mutual
              << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Check_source_target: " << std::boolalpha << check_source_target
              << cpp_tools::colors::reset << '\n';

    // to check when source != target
    const int dimension = 2;

    if(check_source_target)
    {
        input_target_file = std::string("../data/sources_targets/circle-100_target.fma");
        input_source_file = std::string("../data/sources_targets/circle-100_source.fma");
        std::cout << cpp_tools::colors::blue << "<params> Sources file: " << input_source_file
                  << cpp_tools::colors::reset << '\n';
        std::cout << cpp_tools::colors::blue << "<params> Targets file: " << input_target_file
                  << cpp_tools::colors::reset << '\n';
    }
    else
    {
        input_target_file = std::string("../data/sources_targets/circle-100_target.fma");
        input_source_file = std::string("../data/sources_targets/circle-100_target.fma");
        std::cout << cpp_tools::colors::blue << "<params> Sources=Target file: " << input_source_file
                  << cpp_tools::colors::reset << '\n';
    }
    //
    const int matrix_type = 0;
    switch(matrix_type)
    {
    case 0:
        using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
        //
        using options = scalfmm::options::uniform_<scalfmm::options::fft_>;

        if(dimension == 2)
        {
            using interpolation_type =
              scalfmm::interpolation::interpolator<value_type, 2, far_matrix_kernel_type, options>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

            using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;
            fmm_run<2, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
                                                       order, check_source_target, mutual);
        }
        else
        {
            using interpolation_type =
              scalfmm::interpolation::interpolator<value_type, 3, far_matrix_kernel_type, options>;
            using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;

            using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;
            fmm_run<3, value_type, fmm_operators_type>(input_source_file, input_target_file, tree_height, group_size,
                                                       order, check_source_target, mutual);
        }
        break;
    default:
        std::cout << "Kernel not implemented. values are\n Laplace kernels: 0) 1/r, 1) grad(1/r),"
                  << " 2) p + grad(1/r) 3) like_mrhs." << std::endl
                  << "Scalar kernels 4) 1/r^2 5) ln in 2d" << std::endl;
    }
}
