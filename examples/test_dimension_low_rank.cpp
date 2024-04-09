#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/options/options.hpp"
#include "scalfmm/utils/parameters.hpp"

#include "scalfmm/algorithms/fmm.hpp"

#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"

#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"

#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/sort.hpp"
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// example:
// ./examples/RelWithDebInfo/test_dimension --tree-height  4 -gs 2 --order 4 --dimension 3
namespace local_args
{
    struct isSorted
    {
        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--data-sorted", "--ds"};
        std::string description = "Precise if the data are sorted by their morton index";
    };

    struct dimension
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n   2 for dimension 2, 3 for dimension 3";
        using type = int;
        type def = 1;
    };

    struct Notmutual
    {
        cpp_tools::cl_parser::str_vec flags = {"--not-mutual"};
        std::string description = "Do not consider dependencies in p2p interaction list (not mutual algorithm)";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };

}   // namespace local_args

template<int Dimension, typename CONTAINER_T, typename POINT_T, typename VALUE_T>
void read_data(const std::string& filename, CONTAINER_T*& container, POINT_T& Centre, VALUE_T& width)
{
    //  std::cout << "READ DATA " << std::endl << std::flush;
    using particle_type = typename CONTAINER_T::particle_type;
    using value_type = typename POINT_T::value_type;
    bool verbose = true;

    scalfmm::io::FFmaGenericLoader<value_type, Dimension> loader(filename, verbose);

    const int number_of_particles = loader.getNumberOfParticles();
    width = loader.getBoxWidth();
    Centre = loader.getBoxCenter();

    auto nb_val_to_red_per_part = loader.getNbRecordPerline();
    // could be a problem for binary file (float double)
    value_type* values_to_read = new value_type[nb_val_to_red_per_part]{0};
    container = new CONTAINER_T(number_of_particles);
    std::cout << "number_of_particles " << number_of_particles << std::endl;
    for(std::size_t idx = 0; idx < number_of_particles; ++idx)
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
        p.variables(idx, values_to_read[ii++]);
        container->insert_particle(idx, p);
    }
}

template<typename Tree_T, typename Container_T, typename Value_T>
auto inline check_results(const Tree_T& tree, const Container_T& reference, const Value_T& eps) -> bool
{
    scalfmm::utils::accurater<Value_T> error;

    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&reference, &error](auto const& leaf)
                                      {
                                          const auto nb_elt = leaf.size();
                                          for(std::size_t i = 0; i < nb_elt; ++i)
                                          {
                                              const auto& p = leaf.particle(i);
                                              std::cout << p << std::endl;
                                              const auto& idx = std::get<0>(p.variables());
                                              error.add(std::get<0>(p.outputs()),
                                                        std::get<1>(reference.variables(idx)));
                                          }
                                      });
    bool ok = (error.get_relative_l2_norm() < eps);
    std::cout << "Error " << error << std::endl;
    return ok;
}

template<int Dimension, typename value_type, class FMM_OPERATOR_TYPE>
auto run(const std::string& input_file, const int& tree_height, const int& group_size, const int& order,
         const bool mutual) -> int
{
    using near_matrix_kernel_type = typename FMM_OPERATOR_TYPE::near_field_type::matrix_kernel_type;
    using far_field_type = typename FMM_OPERATOR_TYPE::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;

    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;
    //
    //
    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    //  The matrix kernel
    //
    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    std::cout << cpp_tools::colors::green << "Tree parameters\n"
              << "   tree_height: " << tree_height << std::endl
              << "   group_size:  " << group_size << std::endl
              << "   order:       " << order << std::endl
              << cpp_tools::colors::reset;
    // Open particle file
    cpp_tools::timers::timer time{};

    // ---------------------------------------
    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, int, value_type>;

    using container_type = scalfmm::container::particle_container<particle_type>;
    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "Creating & Inserting particles ...\n" << cpp_tools::colors::reset;

    scalfmm::container::point<value_type, Dimension> box_center{};
    value_type box_width{};

    time.tic();
    container_type* container{};
    read_data<Dimension>(input_file, container, box_center, box_width);
    time.tac();

    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << " box width = " << box_width
              << cpp_tools::colors::reset << '\n';

    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    time.tic();
    box_type box(box_width, box_center);
    group_tree_type tree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                         static_cast<std::size_t>(group_size), *container);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Group tree created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    time.tic();
    far_matrix_kernel_type mk_far{};
    interpolator_type interpolator(mk_far, order, static_cast<std::size_t>(tree_height), box.width(0));
    near_matrix_kernel_type mk_near{};
    typename FMM_OPERATOR_TYPE::near_field_type near_field(mk_near, mutual);
    typename FMM_OPERATOR_TYPE::far_field_type far_field(interpolator);
    FMM_OPERATOR_TYPE fmm_operator(near_field, far_field);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Kernel and Interp created in " << time.elapsed() << "ms\n"
              << cpp_tools::colors::reset;

    auto const separation_criterion = fmm_operator.near_field().separation_criterion();
    time.tic();
    scalfmm::list::sequential::build_interaction_lists(tree, tree, separation_criterion, mutual);
    time.tac();
    std::cout << cpp_tools::colors::yellow << "Interaction list built in " << time.elapsed() << " ms\n"
              << cpp_tools::colors::reset;

    auto operator_to_proceed = scalfmm::algorithms::all;

    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::timit)](tree, fmm_operator, operator_to_proceed);

    value_type eps = std::pow(10.0, 1 - order);
    bool works = check_results(tree, *container, eps);
    if(works)
    {
        std::cout << cpp_tools::colors::blue << " Test Ok \n" << cpp_tools::colors::reset;
    }
    else
    {
        std::cout << cpp_tools::colors::red << " Test is WRONG !! the error must be around  " << eps << " \n "
                  << cpp_tools::colors::reset;
    }
    delete container;
    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    using value_type = double;

    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, args::tree_height{}, args::order{}, args::input_file{}, args::block_size{},
      args::log_file{}, args::log_level{}, local_args::dimension{}, local_args::Notmutual{});
    parser.parse(argc, argv);
    // Getting command line parameters
    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';

    const int group_size{parser.get<args::block_size>()};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';

    const auto order{parser.get<args::order>()};
    const int dimension = parser.get<local_args::dimension>();
    //
    const std::string input_file(parser.get<args::input_file>());

    const bool mutual{!parser.exists<local_args::Notmutual>()};

    switch(dimension)
    {
    case 1:
    {
        constexpr int dim = 1;
        if(!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                      << '\n';
        }
        using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = scalfmm::interpolation::chebyshev_interpolator<value_type, dim, matrix_kernel_type,
                                                                                 scalfmm::options::low_rank_>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
          input_file, tree_height, group_size, order, mutual);
        break;
    }
    case 2:
    {
        constexpr int dim = 2;
        if(!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                      << '\n';
        }
        using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = scalfmm::interpolation::uniform_interpolator<value_type, dim, matrix_kernel_type,
                                                                               scalfmm::options::low_rank_>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
          input_file, tree_height, group_size, order, mutual);
        break;
    }
    case 3:
    {
        constexpr int dim = 3;

        if(!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                      << '\n';
        }
        using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = scalfmm::interpolation::chebyshev_interpolator<value_type, dim, matrix_kernel_type,
                                                                                 scalfmm::options::low_rank_>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
          input_file, tree_height, group_size, order, mutual);
        break;
    }
    case 4:
    {
        constexpr int dim = 4;

        if(!input_file.empty())
        {
            std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                      << '\n';
        }
        using matrix_kernel_type = scalfmm::matrix_kernels::others::one_over_r2;
        // using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
        //
        using interpolator_type = scalfmm::interpolation::uniform_interpolator<value_type, dim, matrix_kernel_type,
                                                                               scalfmm::options::low_rank_>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;

        run<dim, value_type, scalfmm::operators::fmm_operators<near_field_type, far_field_type>>(
          input_file, tree_height, group_size, order, mutual);
        break;
    }
    default:
        std::cout << "check  1/r^2 Kernel for dimension 2 and 3. Value is \n"
                  << "          0 for dimension 2 "
                  << "          1 for dimension 3 " << std::endl;
        break;
    }
}
