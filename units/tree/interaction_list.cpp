//
// Units for containers
// --------------------
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/tree/for_each.hpp"
#include <cmath>
#define CATCH_CONFIG_RUNNER
#include <algorithm>
#include <catch2/catch.hpp>
#include <string>

//#define SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT
//#include <scalfmm/utils/static_assert_as_exception.hpp>
// Followed by file where assertions must be tested
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/tensor.hpp"

auto get_symbolic_list_p2p(std::size_t morton)
{
    using array_type = std::array<std::size_t, 9>;
    switch(morton)
    {
    case 0:
        return array_type{1, 2, 3};
    case 1:
        return array_type{0, 2, 3, 4, 6};
    case 2:
        return array_type{0, 1, 3, 8, 9};
    case 3:
        return array_type{0, 1, 2, 4, 6, 8, 9, 12};
    case 4:
        return array_type{1, 3, 5, 6, 7};
    case 5:
        return array_type{4, 6, 7, 16, 18};
    case 6:
        return array_type{1, 3, 4, 5, 7, 9, 12, 13};
    case 7:
        return array_type{4, 5, 6, 12, 13, 16, 18, 24};
    case 8:
        return array_type{2, 3, 9, 10, 11};
    case 9:
        return array_type{2, 3, 6, 8, 10, 11, 12, 14};
    case 10:
        return array_type{8, 9, 11, 32, 33};
    case 11:
        return array_type{8, 9, 10, 12, 14, 32, 33, 36};
    case 12:
        return array_type{3, 6, 7, 9, 11, 13, 14, 15};
    case 13:
        return array_type{6, 7, 12, 14, 15, 18, 24, 26};
    case 14:
        return array_type{9, 11, 12, 13, 15, 33, 36, 37};
    case 15:
        return array_type{12, 13, 14, 24, 26, 36, 37, 48};
    case 16:
        return array_type{5, 7, 17, 18, 19};
    case 17:
        return array_type{16, 18, 19, 20, 22};
    case 18:
        return array_type{5, 7, 13, 16, 17, 19, 24, 25};
    case 19:
        return array_type{16, 17, 18, 20, 22, 24, 25, 28};
    case 20:
        return array_type{17, 19, 21, 22, 23};
    case 21:
        return array_type{20, 22, 23};
    case 22:
        return array_type{17, 19, 20, 21, 23, 25, 28, 29};
    case 23:
        return array_type{20, 21, 22, 28, 29};
    case 24:
        return array_type{7, 13, 15, 18, 19, 25, 26, 27};
    case 25:
        return array_type{18, 19, 22, 24, 26, 27, 28, 30};
    case 26:
        return array_type{13, 15, 24, 25, 27, 37, 48, 49};
    case 27:
        return array_type{24, 25, 26, 28, 30, 48, 49, 52};
    case 28:
        return array_type{19, 22, 23, 25, 27, 29, 30, 31};
    case 29:
        return array_type{22, 23, 28, 30, 31};
    case 30:
        return array_type{25, 27, 28, 29, 31, 49, 52, 53};
    case 31:
        return array_type{28, 29, 30, 52, 53};
    case 32:
        return array_type{10, 11, 33, 34, 35};
    case 33:
        return array_type{10, 11, 14, 32, 34, 35, 36, 38};
    case 34:
        return array_type{32, 33, 35, 40, 41};
    case 35:
        return array_type{32, 33, 34, 36, 38, 40, 41, 44};
    case 36:
        return array_type{11, 14, 15, 33, 35, 37, 38, 39};
    case 37:
        return array_type{14, 15, 26, 36, 38, 39, 48, 50};
    case 38:
        return array_type{33, 35, 36, 37, 39, 41, 44, 45};
    case 39:
        return array_type{36, 37, 38, 44, 45, 48, 50, 56};
    case 40:
        return array_type{34, 35, 41, 42, 43};
    case 41:
        return array_type{34, 35, 38, 40, 42, 43, 44, 46};
    case 42:
        return array_type{40, 41, 43};
    case 43:
        return array_type{40, 41, 42, 44, 46};
    case 44:
        return array_type{35, 38, 39, 41, 43, 45, 46, 47};
    case 45:
        return array_type{38, 39, 44, 46, 47, 50, 56, 58};
    case 46:
        return array_type{41, 43, 44, 45, 47};
    case 47:
        return array_type{44, 45, 46, 56, 58};
    case 48:
        return array_type{15, 26, 27, 37, 39, 49, 50, 51};
    case 49:
        return array_type{26, 27, 30, 48, 50, 51, 52, 54};
    case 50:
        return array_type{37, 39, 45, 48, 49, 51, 56, 57};
    case 51:
        return array_type{48, 49, 50, 52, 54, 56, 57, 60};
    case 52:
        return array_type{27, 30, 31, 49, 51, 53, 54, 55};
    case 53:
        return array_type{30, 31, 52, 54, 55};
    case 54:
        return array_type{49, 51, 52, 53, 55, 57, 60, 61};
    case 55:
        return array_type{52, 53, 54, 60, 61};
    case 56:
        return array_type{39, 45, 47, 50, 51, 57, 58, 59};
    case 57:
        return array_type{50, 51, 54, 56, 58, 59, 60, 62};
    case 58:
        return array_type{45, 47, 56, 57, 59};
    case 59:
        return array_type{56, 57, 58, 60, 62};
    case 60:
        return array_type{51, 54, 55, 57, 59, 61, 62, 63};
    case 61:
        return array_type{54, 55, 60, 62, 63};
    case 62:
        return array_type{57, 59, 60, 61, 63};
    case 63:
        return array_type{60, 61, 62};
    default:
        return array_type{};
    }
}
auto get_symbolic_list_m2l(std::size_t morton)
{
    using array_type = std::array<std::size_t, 27>;
    switch(morton)
    {
    case 0:
        return array_type{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    case 1:
        return array_type{5, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    case 2:
        return array_type{4, 5, 6, 7, 10, 11, 12, 13, 14, 15};
    case 3:
        return array_type{5, 7, 10, 11, 13, 14, 15};
    case 4:
        return array_type{0, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27};
    case 5:
        return array_type{0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 24, 25, 26, 27};
    case 6:
        return array_type{0, 2, 8, 10, 11, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27};
    case 7:
        return array_type{0, 1, 2, 3, 8, 9, 10, 11, 14, 15, 17, 19, 25, 26, 27};
    case 8:
        return array_type{0, 1, 4, 5, 6, 7, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39};
    case 9:
        return array_type{0, 1, 4, 5, 7, 13, 15, 32, 33, 34, 35, 36, 37, 38, 39};
    case 10:
        return array_type{0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 34, 35, 36, 37, 38, 39};
    case 11:
        return array_type{0, 1, 2, 3, 4, 5, 6, 7, 13, 15, 34, 35, 37, 38, 39};
    case 12:
        return array_type{0,  1,  2,  4,  5,  8,  10, 16, 17, 18, 19, 24, 25, 26,
                          27, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51};
    case 13:
        return array_type{0,  1,  2,  3,  4,  5,  8,  9,  10, 11, 16, 17, 19, 25,
                          27, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51};
    case 14:
        return array_type{0,  1,  2,  3,  4,  5,  6,  7,  8,  10, 16, 17, 18, 19,
                          24, 25, 26, 27, 32, 34, 35, 38, 39, 48, 49, 50, 51};
    case 15:
        return array_type{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17,
                          18, 19, 25, 27, 32, 33, 34, 35, 38, 39, 49, 50, 51};
    case 16:
        return array_type{4, 6, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    case 17:
        return array_type{4, 5, 6, 7, 12, 13, 14, 15, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    case 18:
        return array_type{4, 6, 12, 14, 15, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31};
    case 19:
        return array_type{4, 5, 6, 7, 12, 13, 14, 15, 21, 23, 26, 27, 29, 30, 31};
    case 20:
        return array_type{16, 18, 24, 25, 26, 27, 28, 29, 30, 31};
    case 21:
        return array_type{16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31};
    case 22:
        return array_type{16, 18, 24, 26, 27, 30, 31};
    case 23:
        return array_type{16, 17, 18, 19, 24, 25, 26, 27, 30, 31};
    case 24:
        return array_type{4,  5,  6,  12, 14, 16, 17, 20, 21, 22, 23, 28, 29, 30,
                          31, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55};
    case 25:
        return array_type{4,  5,  6,  7,  12, 13, 14, 15, 16, 17, 20, 21, 23, 29,
                          31, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55};
    case 26:
        return array_type{4,  5,  6,  7,  12, 14, 16, 17, 18, 19, 20, 21, 22, 23,
                          28, 29, 30, 31, 36, 38, 39, 50, 51, 52, 53, 54, 55};
    case 27:
        return array_type{4,  5,  6,  7,  12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                          22, 23, 29, 31, 36, 37, 38, 39, 50, 51, 53, 54, 55};
    case 28:
        return array_type{16, 17, 18, 20, 21, 24, 26, 48, 49, 50, 51, 52, 53, 54, 55};
    case 29:
        return array_type{16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 48, 49, 50, 51, 52, 53, 54, 55};
    case 30:
        return array_type{16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 48, 50, 51, 54, 55};
    case 31:
        return array_type{16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 48, 49, 50, 51, 54, 55};
    case 32:
        return array_type{8, 9, 12, 13, 14, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    case 33:
        return array_type{8, 9, 12, 13, 15, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    case 34:
        return array_type{8, 9, 10, 11, 12, 13, 14, 15, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47};
    case 35:
        return array_type{8, 9, 10, 11, 12, 13, 14, 15, 37, 39, 42, 43, 45, 46, 47};
    case 36:
        return array_type{8,  9,  10, 12, 13, 24, 25, 26, 27, 32, 34, 40, 41, 42,
                          43, 44, 45, 46, 47, 48, 49, 50, 51, 56, 57, 58, 59};
    case 37:
        return array_type{8,  9,  10, 11, 12, 13, 24, 25, 27, 32, 33, 34, 35, 40,
                          41, 42, 43, 44, 45, 46, 47, 49, 51, 56, 57, 58, 59};
    case 38:
        return array_type{8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 32, 34,
                          40, 42, 43, 46, 47, 48, 49, 50, 51, 56, 57, 58, 59};
    case 39:
        return array_type{8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 32, 33,
                          34, 35, 40, 41, 42, 43, 46, 47, 49, 51, 57, 58, 59};
    case 40:
        return array_type{32, 33, 36, 37, 38, 39, 44, 45, 46, 47};
    case 41:
        return array_type{32, 33, 36, 37, 39, 45, 47};
    case 42:
        return array_type{32, 33, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47};
    case 43:
        return array_type{32, 33, 34, 35, 36, 37, 38, 39, 45, 47};
    case 44:
        return array_type{32, 33, 34, 36, 37, 40, 42, 48, 49, 50, 51, 56, 57, 58, 59};
    case 45:
        return array_type{32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 48, 49, 51, 57, 59};
    case 46:
        return array_type{32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 48, 49, 50, 51, 56, 57, 58, 59};
    case 47:
        return array_type{32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 51, 57, 59};
    case 48:
        return array_type{12, 13, 14, 24, 25, 28, 29, 30, 31, 36, 38, 44, 45, 46,
                          47, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    case 49:
        return array_type{12, 13, 14, 15, 24, 25, 28, 29, 31, 36, 37, 38, 39, 44,
                          45, 46, 47, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    case 50:
        return array_type{12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 36, 38,
                          44, 46, 47, 52, 53, 54, 55, 58, 59, 60, 61, 62, 63};
    case 51:
        return array_type{12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 36, 37,
                          38, 39, 44, 45, 46, 47, 53, 55, 58, 59, 61, 62, 63};
    case 52:
        return array_type{24, 25, 26, 28, 29, 48, 50, 56, 57, 58, 59, 60, 61, 62, 63};
    case 53:
        return array_type{24, 25, 26, 27, 28, 29, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63};
    case 54:
        return array_type{24, 25, 26, 27, 28, 29, 30, 31, 48, 50, 56, 58, 59, 62, 63};
    case 55:
        return array_type{24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 56, 57, 58, 59, 62, 63};
    case 56:
        return array_type{36, 37, 38, 44, 46, 48, 49, 52, 53, 54, 55, 60, 61, 62, 63};
    case 57:
        return array_type{36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 52, 53, 55, 61, 63};
    case 58:
        return array_type{36, 37, 38, 39, 44, 46, 48, 49, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63};
    case 59:
        return array_type{36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 61, 63};
    case 60:
        return array_type{48, 49, 50, 52, 53, 56, 58};
    case 61:
        return array_type{48, 49, 50, 51, 52, 53, 56, 57, 58, 59};
    case 62:
        return array_type{48, 49, 50, 51, 52, 53, 54, 55, 56, 58};
    case 63:
        return array_type{48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
    default:
        return array_type{};
    }
}

TEST_CASE("interaction list p2p", "[interaction-list-p2p]")
{
    using namespace scalfmm;
    constexpr std::size_t Dimension = 2;
    using value_type = double;
    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

    static constexpr std::size_t nb_inputs_near{matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{matrix_kernel_type::kn};
    static constexpr std::size_t nb_inputs_far{matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_far{matrix_kernel_type::kn};

    using particle_type =
      scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type, nb_outputs_near>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type =
      scalfmm::component::cell<scalfmm::component::grid_storage<value_type, Dimension, nb_inputs_far, nb_outputs_far>>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    // generate particles: one par leaf, the octree is full.

    const std::size_t tree_height{4};
    const std::size_t order{3};
    const double box_width{1.};
    const double step{box_width / std::pow(2, (tree_height - 1))};
    const scalfmm::container::point<double, Dimension> box_center(0.);

    const auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));

    container_type* container;
    xt::xarray<std::tuple<double, double, double, double>> particles(
      std::vector(Dimension, number_of_values_per_dimension));

    auto particle_generator = scalfmm::tensor::generate_meshgrid<Dimension>(xt::linspace(
      double(-box_width / 2.) + step * 0.5, double(box_width / 2.) - step * 0.5, number_of_values_per_dimension));
    auto eval_generator = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::eval(std::forward<decltype(xs)>(xs))...); }, particle_generator);
    auto flatten_views = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::flatten(std::forward<decltype(xs)>(xs))...); }, eval_generator);

    auto particle_flatten_views = xt::flatten(particles);
    container = new container_type(particles.size());
    auto container_it = std::begin(*container);
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        *container_it = std::apply(
          [&i](auto&&... xs) { return std::make_tuple(std::forward<decltype(xs)>(xs)[i]..., 0., 0.); }, flatten_views);
        ++container_it;
    }

    box_type box(box_width, box_center);

    SECTION("2D, group-size = 8", "[2D-gs=8]")
    {
        const std::size_t group_size{8};
        group_tree_type tree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                             static_cast<std::size_t>(group_size), *container);

        const int neighbour_separation = 1;
        const bool with_depend = true;
       scalfmm::list::sequential::build_interaction_lists( tree, tree, neighbour_separation, with_depend);
           if(with_depend)
    {
        scalfmm::list::reconstruct_p2p_mutual_interaction_lists(tree);
    }
        tree.trace(std::cout, 4);
        scalfmm::component::for_each_leaf(
          tree.begin(), tree.end(),
          [](auto const& leaf)
          {
              auto const& indexes = leaf.csymbolics().interaction_indexes;
              auto interaction_morton_index = leaf.csymbolics().interaction_iterators;
                        auto my_index = leaf.index();

              auto ref1{get_symbolic_list_p2p(leaf.index())};
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

          for(std::size_t i{0}; i < leaf.csymbolics().existing_neighbors_in_group; ++i)
          {
                   REQUIRE(ref.at(i) == indexes.at(i));
              }
              for(std::size_t i{0}; i < leaf.csymbolics().existing_neighbors_in_group; ++i)
              {
                  REQUIRE(interaction_morton_index.at(i)->index() ==
                          *std::find(std::begin(indexes), std::end(indexes), interaction_morton_index.at(i)->index()));
              }
          });
        scalfmm::component::for_each_cell(tree.begin(), tree.end(), 3,
                                          [](auto const& cell)
                                          {
                                              auto const& indexes = cell.csymbolics().interaction_indexes;
                                              auto interaction_morton_index = cell.csymbolics().interaction_iterators;
                                              //   auto interaction_positions = cell.csymbolics().interaction_positions;
                                              auto ref{get_symbolic_list_m2l(cell.index())};
                                              for(std::size_t i{0}; i < cell.csymbolics().number_of_neighbors; ++i)
                                              {
                                                  REQUIRE(ref.at(i) == indexes.at(i));
                                              }
                                              for(std::size_t i{0}; i < cell.csymbolics().existing_neighbors; ++i)
                                              {
                                                  REQUIRE(ref.at(i) == interaction_morton_index.at(i)->index());
                                              }
                                          });
    }

    SECTION("2D, group-size = 1", "[2D-gs=1]")
    {
        const std::size_t group_size{1};
        group_tree_type tree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                             static_cast<std::size_t>(group_size), *container);

        const int neighbour_separation = 1;
        const bool with_depend = false;
        scalfmm::list::sequential::build_interaction_lists( tree, tree, neighbour_separation, with_depend);

    if(with_depend)
    {
        scalfmm::list::reconstruct_p2p_mutual_interaction_lists(tree);
    }
        scalfmm::component::for_each_leaf(
          tree.begin(), tree.end(),
          [](auto const& leaf)
          {
              auto const& indexes = leaf.csymbolics().interaction_indexes;
              auto interaction_morton_index = leaf.csymbolics().interaction_iterators;
              auto ref{get_symbolic_list_p2p(leaf.index())};
              for(std::size_t i{0}; i < leaf.csymbolics().number_of_neighbors; ++i)
              {
                  REQUIRE(ref.at(i) == indexes.at(i));
              }
              for(std::size_t i{0}; i < leaf.csymbolics().existing_neighbors_in_group; ++i)
              {
                  REQUIRE(interaction_morton_index.at(i)->index() ==
                          *std::find(std::begin(indexes), std::end(indexes), interaction_morton_index.at(i)->index()));
              }
          });

        scalfmm::component::for_each_cell(tree.begin(), tree.end(), 3,
                                          [](auto const& cell)
                                          {
                                              auto const& indexes = cell.csymbolics().interaction_indexes;
                                              auto interaction_morton_index = cell.csymbolics().interaction_iterators;
                                              //   auto interaction_positions = cell.csymbolics().interaction_positions;
                                              auto ref{get_symbolic_list_m2l(cell.index())};
                                              for(std::size_t i{0}; i < cell.csymbolics().number_of_neighbors; ++i)
                                              {
                                                  REQUIRE(ref.at(i) == indexes.at(i));
                                              }
                                              for(std::size_t i{0}; i < cell.csymbolics().existing_neighbors; ++i)
                                              {
                                                  REQUIRE(ref.at(i) == interaction_morton_index.at(i)->index());
                                              }
                                          });
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
