

#define CHECK_ADL_ON_P2M
// #define CHECK_ADL_ON_P2M_WITH_USING   // pas d'ADL
// #define CHECK_ADL_ON_P2P
// #define CHECK_ADL_ON_P2P_WITH_USING

#include "in_check_adl.hpp"

#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tree/tree.hpp"

#include <random>
#include <vector>
//
//  ./checks/Release/check_adl
//
int main()
{
    static constexpr int dimension = 3;
    using value_type = double;
#ifdef CHECK_ADL_ON_P2P
    using near_matrix_kernel_type = check_adl::operators::empty_kernel;
    using near_field_type = check_adl::operators::near_field_operator_type;
#else
    using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
#endif

    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    using options = scalfmm::options::uniform_<scalfmm::options::fft_>;
    using interpolation_type =
      scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, options>;
#ifdef CHECK_ADL_ON_P2M
    using far_field_type = check_adl::operators::new_far_field_operator<interpolation_type, false>;
#else
    using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, false>;
#endif

    using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    static constexpr std::size_t nb_inputs{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs{near_matrix_kernel_type::kn};
    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs>;

    using container_type = std::vector<particle_type>;
    using position_type = typename particle_type::position_type;
    using box_type = scalfmm::component::box<position_type>;
    // tree types
    using cell_type = scalfmm::component::cell<typename interpolation_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;
    /////////////////////////////////////
    ////
    const std::size_t nb_particles{100};
    const value_type box_width{2.};
    const position_type box_center(1.);
    box_type box(box_width, box_center);
    container_type container(nb_particles);
    //

    std::mt19937 gen(123);
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
    }
    const std::size_t tree_height{3};
    const std::size_t group_size{10};   // the number of cells and leaf grouped in the tree
    const std::size_t order{4};

    tree_type tree(tree_height, order, box, group_size, group_size, container);
    //
    //////////////////////////////////////////
    // ffm operator
    near_matrix_kernel_type mk_near{};
    const bool mutual_near = false;
    near_field_type near_field(mk_near, mutual_near);

    far_matrix_kernel_type mk_far{};
    interpolation_type interpolator(mk_far, order, tree_height, box_width);
    far_field_type far_field(interpolator);

    fmm_operators_type fmm_operator(near_field, far_field);

    std::cout << cpp_tools::colors::blue;
    fmm_operator.settings(std::cout);
    std::cout << cpp_tools::colors::reset;
    //
    //////////////////////////////////////////
    // Algorithm

    auto operator_to_proceed = scalfmm::algorithms::all;
    //  auto operator_to_proceed = scalfmm::algorithms::nearfield;

    std::cout << cpp_tools::colors::blue << "operator_to_proceed: ";
    scalfmm::algorithms::print(operator_to_proceed);
    std::cout << cpp_tools::colors::reset << std::endl;

    scalfmm::algorithms::sequential::sequential(tree, fmm_operator, operator_to_proceed);

    return 1;
}