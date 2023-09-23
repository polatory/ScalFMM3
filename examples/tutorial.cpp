#include <random>
#include <vector>

#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/accurater.hpp"

#include "scalfmm/meta/utils.hpp"

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // order of the approximation
    const std::size_t order{5};
    // height of the fmm tree
    const std::size_t tree_height{5};

    using namespace scalfmm;
    using value_type = double;

    // choosing a matrix kernel
    // the far field matrix kernel
    using far_matrix_kernel_type = matrix_kernels::laplace::one_over_r;
    // the near field matrix kernel
    using near_matrix_kernel_type = far_matrix_kernel_type;
    // number of inputs and outputs.
    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};

    // loading data in containers
    static constexpr std::size_t dimension{2};

    using particle_type = container::particle<
      // position
      value_type, dimension,
      // inputs
      value_type, nb_inputs_near,
      // outputs
      value_type, nb_outputs_near,
      // variables
      std::size_t   // for storing the index in the original container.
      >;
    // position point type
    using position_type = typename particle_type::position_type;

    using container_type = std::vector<particle_type>;
    // allocate 100 particles.
    constexpr std::size_t nb_particles{100};
    container_type container(nb_particles);

    // box of the simulation [0,2]x[0,2]
    using box_type = component::box<position_type>;
    // width of the box
    constexpr value_type box_width{2.};
    // center of the box
    const position_type box_center{1., 1.};
    // the box for the tree
    box_type box(box_width, box_center);

    // random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<value_type> dis(0.0, 2.0);
    auto random_r = [&dis, &gen]() { return dis(gen); };

    // inserting particles in the container
    for(std::size_t idx = 0; idx < nb_particles; ++idx)
    {
        // particle_type p;
        particle_type& p = container[idx];
        for(auto& e: p.position())
        {
            e = random_r();
        }
        for(auto& e: p.inputs())
        {
            e = random_r();
        }
        for(auto& e: p.outputs())
        {
            e = value_type(0.);
        }
        p.variables(idx);
    }

    // interpolation types
    // we define a near_field from its matrix kernel
    using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
    // we choose an interpolator with a far matrix kernel for the approximation
    using interpolator_type = interpolation::interpolator<value_type, dimension, far_matrix_kernel_type,
                                                          options::uniform_<options::fft_>>;
    // then, we define the far field
    using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
    // the resulting fmm operator is
    using fmm_operator_type = operators::fmm_operators<near_field_type, far_field_type>;
    // construct the fmm operator 
    // construct the near field
    near_field_type near_field;
    // a reference on the matrix_kernel of the near_field
    auto& near_mk = near_field.matrix_kernel();
    // build the approximation used in the near field
    interpolator_type interpolator(order, tree_height, box.width(0));
    far_field_type far_field(interpolator);
    //  construct the fmm operator 
    fmm_operator_type fmm_operator(near_field, far_field);

    // tree types
    // the cell type of the tree holding multipoles and locals expansions
    // here, we extract the correct storage for the cells from the interpolation method.
    using cell_type = component::cell<typename interpolator_type::storage_type>;
    // the leaf type holding the particles
    using leaf_type = component::leaf_view<particle_type>;
    // the tree type
    using group_tree_type = component::group_tree_view<cell_type, leaf_type, box_type>;
    // we construct the tree
    const std::size_t group_size{10};   // the number of cells and leaf grouped in the tree
    group_tree_type tree(tree_height, order, box, group_size, group_size, container);

    // now we have everything to call the fmm algorithm
    algorithms::fmm[options::_s(options::seq)](tree, fmm_operator);

    // we will compute the reference with the full direct algorithm
    // from the original container

    algorithms::full_direct(container, near_mk);

    utils::accurater<value_type> error;

    component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                             [&container, &error](auto const& leaf)
                             {
                                 // loop on the particles of the leaf
                                 for(auto const p_ref: leaf)
                                 {
                                     // build a particle
                                     const auto p = typename leaf_type::const_proxy_type(p_ref);
                                     //
                                     const auto& idx = std::get<0>(p.variables());

                                     auto const& output_ref = container[idx].outputs();
                                     auto const& output = p.outputs();

                                     for(int i{0}; i < nb_outputs_near; ++i)
                                     {
                                         error.add(output_ref.at(i), output.at(i));
                                     }
                                 }
                             });

    std::cout << error << '\n';

    return 0;
}
