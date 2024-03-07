// @FUSE_OMP
/**
 * @file test-rand-3.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-10
 *
 * @copyright Copyright (c) 2023
 *
 */

/**
 *\code
 * check/Release/check_source_target -th 3 -gs 10  --order 6 --scale 1.0,1.0,1.0
 *\endcode
 */
//
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <valarray>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"

#include "xtensor-blas/xblas.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor_forward.hpp>
//
//
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/algorithms/omp/utils.hpp"
#include "scalfmm/algorithms/sequential/utils.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
//
// Tree
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/tree/utils.hpp"
//
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include "scalfmm/utils/source_target.hpp"
//
#include "scalfmm/utils/parameters.hpp"

#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

// using namespace scalfmm::io;

//
namespace local_args
{
    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
    };
    struct ns
    {
        cpp_tools::cl_parser::str_vec flags = {"--number_sources", "-ns"};
        std::string description = "Number of source particles.";
        using type = int;
        type def = 2000000;
    };
    struct nt
    {
        cpp_tools::cl_parser::str_vec flags = {"--number_targets", "-nt"};
        std::string description = "Number of target particles.";
        using type = int;
        type def = 200000;
    };
    struct seed
    {
        cpp_tools::cl_parser::str_vec flags = {"-seed"};
        std::string description = "Seed to generate random particles.";
        using type = int;
        type def = 1323;
    };
    struct scale
    {
        cpp_tools::cl_parser::str_vec flags = {"--scale", "-s"}; /*!< The flags */
        std::string description = "Scale parameter sx,s,sz";     /*!< The description */
        std::string input_hint = "value,value,value";            /*!< The description */
        using type = std::vector<double>;
    };
    struct check
    {
        cpp_tools::cl_parser::str_vec flags = {"--check"};
        std::string description = "Check with p2p ";
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
    };
}   // namespace local_args

using timer_type = cpp_tools::timers::timer<std::chrono::milliseconds>;
using timer_acc_type = cpp_tools::timers::timer<std::chrono::microseconds>;

using value_type = double;
constexpr int dimension = 3;
//
// using options = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;
using options = scalfmm::options::uniform_<scalfmm::options::fft_>;
//  near field
using near_matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
// far field
//  Uniform interpolation and fft optimization
#ifdef SAME_MK
using far_matrix_kernel_type = near_matrix_kernel_type;
using interpolation_type = scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, options>;
using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
#else
using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
using interpolation_type = scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, options>;
using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;
#endif
//
using fmm_operators_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

static constexpr std::size_t nb_inputs{1};
static constexpr std::size_t nb_outputs{1 + dimension};
constexpr int nb_output_source{1};          // should be zero
constexpr int nb_input_target{nb_inputs};   // should be zero

using point_type = scalfmm::container::point<value_type, dimension>;
#ifdef SAME_MK
using particle_source_type =
  scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs, std::size_t>;
using particle_target_type = particle_source_type;
#else
using particle_source_type =
  scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_output_source, std::size_t>;
using particle_target_type =
  scalfmm::container::particle<value_type, dimension, value_type, nb_input_target, value_type, nb_outputs, std::size_t>;
#endif
using leaf_source_type = scalfmm::component::leaf_view<particle_source_type>;
using leaf_target_type = scalfmm::component::leaf_view<particle_target_type>;
// this struct is crucial for having the good type for iterator and group types (source) in
// the target leaves and groups
namespace scalfmm::meta
{
    template<>
    struct inject<scalfmm::component::group_of_particles<leaf_target_type, particle_target_type>>
    {
        using type = std::tuple<
          typename scalfmm::component::group_of_particles<leaf_source_type, particle_source_type>::iterator_type,
          scalfmm::component::group_of_particles<leaf_source_type, particle_source_type>>;
    };

}   // namespace scalfmm::meta
template<typename Particle_type>
inline auto operator<<(std::ostream& os, std::vector<Particle_type> const& container) -> std::ostream&
{
    for(std::size_t i{0}; i < container.size(); ++i)
    {
        std::cout << i << " " << container[i] << std::endl;
    }
    return os;
}
template<typename Tree, typename Value_type, typename Vect>
auto sum_output(Tree const& tree, const Value_type& Q, const int iter, Vect& res) -> void
{
    Value_type potential = Value_type(0.);
    Value_type fx = Value_type(0.), fy = Value_type(0.), fz = Value_type(0.);
    //
    timer_acc_type time{};
    time.tic();
    for(auto pg: tree.vector_of_leaf_groups())
    {
        //  std::cout << pg->storage() << "\n";
        auto nb_particles = pg->storage().size();
        auto ptr_pos_x = pg->storage().ptr_on_position();
        auto ptr_pos_y = ptr_pos_x + nb_particles;
        auto ptr_potential = pg->storage().ptr_on_output();
        auto ptr_fx = ptr_potential + nb_particles;
        auto ptr_fy = ptr_fx + nb_particles;
        auto ptr_fz = ptr_fy;
        auto ptr_pos_z = ptr_pos_y;
        if constexpr(dimension == 3)
        {
            ptr_pos_z = ptr_pos_y + nb_particles;
            ptr_fz = ptr_fy + nb_particles;
        }

        // std::cout << "First output: " << ptr_potential << " " << ptr_fx << " " << ptr_fy << " " << ptr_fz << " \n ";
        for(std::size_t i = 0; i < nb_particles; ++i)
        {
            potential += ptr_potential[i];
            fx += ptr_fx[i];
            fy += ptr_fy[i];
            if constexpr(dimension == 3)
            {
                fz += ptr_fz[i];
            }
        }
    }   // end block
    time.tac();
    std::cout << cpp_tools::colors::yellow << "sum_output (pointer on group) " << time.elapsed() << " mus\n"
              << cpp_tools::colors::reset;
    std::cout << "Forces Sum / Q  x = " << fx / Q << " y = " << fy / Q << " z = " << fz / Q << std::endl;
    std::cout << "Potential / Q    = " << potential / Q << std::endl;
    if(iter == 0)
    {
        res[0] = potential / Q;
        res[1] = fx / Q;
        res[2] = fy / Q;
        res[3] = fz / Q;
    }
    std::cout << std::endl;
}
template<typename Tree, typename Value_type, typename Vect>
auto sum_output_1(Tree const& tree, const Value_type& Q, const int iter, Vect& res) -> void
{
    Value_type potential = Value_type(0.);
    Value_type fx = Value_type(0.), fy = Value_type(0.), fz = Value_type(0.);
    //
    timer_acc_type time{};
    time.tic();
    // Loop on the group
    for(auto const pg: tree.group_of_leaves())
    {
        // Loop on the leaf of thz group
        for(auto const& leaf: pg->components())
        {
            // loop on the particle inside the leaf
            for(auto const particle_tuple_ref: leaf)
            {
                auto p = typename Tree::leaf_type::const_proxy_type(particle_tuple_ref);
                auto const& out = p.outputs();
                potential += out[0];
                fx += out[1];
                fy += out[2];
                if constexpr(dimension == 3)
                {
                    fz += out[3];
                }
            }
        }
    }
    time.tac();
    std::cout << cpp_tools::colors::yellow << "sum_output (loop ob group/leaf/part) " << time.elapsed() << " ms\n"
              << cpp_tools::colors::reset;
    std::cout << "Forces Sum / Q  x = " << fx / Q << " y = " << fy / Q << " z = " << fz / Q << std::endl;
    std::cout << "Potential / Q    = " << potential / Q << std::endl;
}
template<typename Tree, typename Value_type, typename Vect>
auto sum_output_2(Tree const& tree, const Value_type& Q, const int iter, Vect& res) -> void
{
    Value_type potential = Value_type(0.);
    Value_type fx = Value_type(0.), fy = Value_type(0.), fz = Value_type(0.);
    timer_acc_type time{};
    time.tic();
    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&potential, &fx, &fy, &fz](auto const& leaf)
                                      {
                                          for(auto const p_tuple_ref: leaf)
                                          {
                                              const auto p = typename Tree::leaf_type::const_proxy_type(p_tuple_ref);

                                              auto const& out = p.outputs();
                                              potential += out[0];
                                              fx += out[1];
                                              fy += out[2];
                                              if constexpr(dimension == 3)
                                              {
                                                  fz += out[3];
                                              }
                                          }
                                      });
    time.tac();
    std::cout << cpp_tools::colors::yellow << "sum_output (for_eaf_leaf) " << time.elapsed() << " mus\n"
              << cpp_tools::colors::reset;
    std::cout << "Forces Sum / Q  x = " << fx / Q << " y = " << fy / Q << " z = " << fz / Q << std::endl;
    std::cout << "Potential / Q    = " << potential / Q << std::endl;
    if(iter == 0)
    {
        res[0] = potential / Q;
        res[1] = fx / Q;
        res[2] = fy / Q;
        res[3] = fz / Q;
    }
    std::cout << std::endl;
}
template<typename ContainerS, typename ContainerT, typename Value_type>
auto build_matrix(ContainerS const& source, ContainerT const& target, const Value_type& Q) -> void
{
    auto ns = int(source.size());   // col
    auto nt = int(target.size());   // row
    auto one = Value_type(1.0);
    std::vector<Value_type> mat(ns * nt), x(ns, Q), y(nt, Value_type(0.));
    int der = 1;
    for(int i = 0; i < nt; ++i)
    {
        auto const& pos = target[i].position();
        // mat[i * ns + i] = Value_type(0.);
        for(int j = 0; j < ns; ++j)
        {
            auto diff = pos - source[j].position();
            auto tmp = diff.norm();
            // row major

            // mat[i * ns + j] = one / tmp);

            mat[i * ns + j] = -diff[der] / (tmp * tmp * tmp);
        }
    }
    // row major
    auto leading = nt;
    cxxblas::gemv(cxxblas::RowMajor, cxxblas::Transpose::NoTrans, ns, nt, value_type(1.0), mat.data(), leading,
                  x.data(), 1, 0.0, y.data(), 1);
    Value_type sum = std::accumulate(y.begin(), y.end(), Value_type(0.0));
    std::cout << "sum: " << sum << std::endl;
}
//
template<int dimension, typename value_type, class fmm_operators_type>
auto fmm_run(const int& tree_height, const int& group_size, const int& order, int& seed,
             const std::vector<value_type>& Scale_v, const int& NbSources, const int NbTargets, bool check_direct,
             const std::string& output_file) -> int
{
    //  The matrix kernel
    using near_matrix_kernel_type = typename fmm_operators_type::near_field_type::matrix_kernel_type;
    using far_field_type = typename fmm_operators_type::far_field_type;
    using interpolator_type = typename far_field_type::approximation_type;
    using far_matrix_kernel_type = typename interpolator_type::matrix_kernel_type;

    timer_type time{};

    // Construct the container of particles
    using container_source_type = scalfmm::container::particle_container<particle_source_type>;
    // using container_target_type = scalfmm::container::particle_container<particle_target_type>;
    using box_type = scalfmm::component::box<point_type>;
    //
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using tree_source_type = scalfmm::component::group_tree_view<cell_type, leaf_source_type, box_type>;
    using tree_target_type = scalfmm::component::group_tree_view<cell_type, leaf_target_type, box_type>;
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "Working on particles ..." << std::endl;
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<value_type> dist(-1, 1);
    seed48(reinterpret_cast<short unsigned int*>(&seed));
    const auto ScaleX = Scale_v[0];
    const auto ScaleY = Scale_v[1];
    const auto ScaleZ = Scale_v[2];
    auto Scale = ScaleX;
    Scale = (ScaleY > Scale ? ScaleY : Scale);
    Scale = (ScaleZ > Scale ? ScaleZ : Scale);
    std::cout << cpp_tools::colors::green << "Creating & Inserting ...\n" << cpp_tools::colors::reset;
    //
    seed48(reinterpret_cast<short unsigned int*>(&seed));

    time.tic();
    point_type box_center_target(0.);
    value_type box_width_target{2.01 * Scale};
    // container_target_type container_target(NbTargets);
    std::vector<particle_target_type> container_target(NbTargets);

    box_type box_target(box_width_target, box_center_target);
    for(int i = 0; i < NbTargets; ++i)
    {   // cibles
        auto& pos = container_target[i].position();
        for(int d = 0; d < dimension; ++d)
        {
            pos[d] = dist(gen) * Scale_v[d];
        }
        container_target[i].inputs(0) = 1.0;
        container_target[i].variables(i);
    }

    point_type box_center_source(0.);
    value_type box_width_source{2.01 * Scale};
    container_source_type container_source(NbSources);
    // std::vector<particle_source_type> container_source(NbSources);

    box_type box_source(box_width_source, box_center_source);
    for(int i = 0; i < NbSources; ++i)
    {   // sources
        scalfmm::meta::repeat([&gen, &dist](auto& v) { v = dist(gen); }, container_source[i].position());
        container_source[i].inputs(0) = 1.0;
        container_source[i].variables(i);
    }

    time.tac();
    std::cout << cpp_tools::colors::yellow << "Container loaded/generated in " << time.elapsed() << " s\n"
              << cpp_tools::colors::reset;
    //
    auto box = scalfmm::utils::bounding_box(box_source, box_target);
    time.tic();
    // build trees
    // We sort the particles to have the same ordering in scalfmm and in the application. This avoids applying the
    // permutation on the inputs and outputs
    constexpr std::size_t max_level = 10;
    scalfmm::utils::sort_container(box, max_level, container_target);
    scalfmm::utils::sort_container(box, max_level, container_source);
    bool sorted = true;
    // the container is now sorted
    tree_source_type tree_source(tree_height, order, box, group_size, group_size, container_source, sorted);
    tree_target_type tree_target(tree_height, order, box, group_size, group_size, container_target, sorted);
    time.tac();

    std::cout << "Done  " << cpp_tools::colors::yellow << "(@Building trees = " << time.elapsed() << " ms)."
              << std::endl;

    std::cout << "bounding_box " << box << std::endl;
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    //
    /////////////////////////////////////////////////////////////////////////////////////
    //
    //              Compute source-target interaction though FMM
    //
    /////////////////////////////////////////////////////////////////////////////////////
    auto box_width = box.width(0);
    // Far field
    far_matrix_kernel_type mk_far{};
    interpolator_type interpolator(mk_far, order, tree_height, box_width);
    typename fmm_operators_type::far_field_type far_field(interpolator);
    // Near field
    near_matrix_kernel_type mk_near{};
    typename fmm_operators_type::near_field_type near_field(mk_near, false /* no mutual for source!= target*/);
    //
    options op;
    std::cout << cpp_tools::colors::blue << "Fmm with kernels: " << std::endl
              << "       near  " << mk_near.name() << std::endl
              << "       far   " << mk_far.name() << std::endl
              << "          option " << op.value() << std::endl;

    fmm_operators_type fmm_operator(near_field, far_field);
    auto neighbour_separation = fmm_operator.near_field().separation_criterion();
    //
    tree_target.build_interaction_lists(tree_source, neighbour_separation, false);

    auto operator_to_proceed = scalfmm::algorithms::all;   // all, nearfield;
    value_type Q{0.0};
    std::valarray<value_type> res(0.0, nb_outputs), error(0.0, nb_outputs);
    for(int iter = 0; iter < 2; ++iter)
    {
        Q = 1.0;   // pow(2.0, double(iter));
        std::cout << "iter = " << iter << " Q = " << Q << std::endl;
        timer_acc_type ttime;
        // Change the input
        // loop on group of leaves
        for(auto pg: tree_source.vector_of_leaf_groups())
        {
            // Get the storage of the inputs (here we only have one input)
            auto input_storage = pg->storage().ptr_on_input();
            for(std::size_t i(0); i < pg->storage().size(); ++i)
            {
                input_storage[i] = Q;
            }
        }
        scalfmm::algorithms::sequential::reset_outputs(tree_target);
        scalfmm::algorithms::sequential::reset_locals(tree_target);
        scalfmm::algorithms::sequential::reset_multipoles(tree_source);
        std::cout << "Start algo\n";

        time.tic();

        scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq_timit)](tree_source, tree_target,
                                                                                    fmm_operator, operator_to_proceed);
        time.tac();
        std::cout << "Done  " << cpp_tools::colors::blue << "(@Algorithm = " << time.elapsed() << " ms)." << std::endl;
        sum_output(tree_target, Q, iter, res);
    }
    // {
    //     int idx_g{0};

    //     auto print_result_to_scalfmm2 = [&idx_g](auto& leaf)
    //     {
    //         auto i{idx_g};
    //         auto inputs_iterator = scalfmm::container::inputs_begin(leaf.cparticles());
    //         auto outputs_iterator = scalfmm::container::outputs_begin(leaf.particles());

    //         for(std::size_t idx = 0; idx < leaf.size(); ++idx, ++inputs_iterator, ++outputs_iterator)
    //         {
    //             // auto const& q = std::get<0>(*inputs_iterator);
    //             auto const q = 3.0;
    //             std::get<0>(*inputs_iterator) /= q;
    //             scalfmm::meta::repeat([q](auto& out) { out *= q; }, *outputs_iterator);
    //         }
    //         i = idx_g;
    //         for(auto const p_tuple_ref: leaf)
    //         {
    //             // We construct a particle type for classical acces
    //             auto proxy = typename tree_target_type::leaf_type::proxy_type(p_tuple_ref);
    //             // std::cout << i++ << " p " << typename tree_target_type::leaf_type::particle_type(p_tuple_ref)
    //             << "
    //             "
    //             //           << std::endl;

    //             auto const& q = proxy.inputs();
    //             // get forces
    //             auto& out = proxy.outputs();
    //             // In ScalFMM2 the forces are multiplied by  q
    //             for(int j = 0; j < tree_target_type::leaf_type::particle_type::outputs_size; ++j)
    //             {
    //                 out[j] *= q[0];
    //             }
    //             // std::cout << idx_g++ << " p " << typename
    //             tree_target_type::leaf_type::particle_type(p_tuple_ref)
    //             <<
    //             // " "
    //             //           << std::endl;
    //         }
    //     };
    //     scalfmm::component::for_each_leaf(std::cbegin(tree_target), std::cend(tree_target),
    //                                       [&idx_g, &print_result_to_scalfmm2](auto const& leaf)
    //                                       { print_result_to_scalfmm2(leaf); });
    // }
    if(check_direct)
    {
        Q = 1;
        std::cout << cpp_tools::colors::green << "full interaction computation  with kernel: " << mk_near.name()
                  << std::endl
                  << cpp_tools::colors::reset;
        time.tic();
        scalfmm::algorithms::full_direct(container_source, container_target, mk_near);
        time.tac();
        std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
        std::cout << cpp_tools::colors::yellow << "Computation done in " << time.elapsed() << " ms\n"
                  << cpp_tools::colors::reset;
        //
        std::valarray<value_type> calc(0.0, nb_outputs);
        // std::cout << "container_target\n" << container_target << std::endl;
        for(std::size_t i{0}; i < container_target.size(); ++i)
        {
            auto output = container_target.at(i).outputs();
            for(std::size_t ii = 0; ii < nb_outputs; ++ii)
            {
                calc[ii] += output.at(ii);
            }
        }
        for(std::size_t i = 0; i < nb_outputs; ++i)
        {
            error[i] = std::abs(calc[i] - res[i]);
        }
        std::cout << "val  = ";
        for(auto v: calc)
        {
            std::cout << " " << v;
        }
        std::cout << std::endl;
        std::cout << "error  = ";
        for(auto v: error)
        {
            std::cout << " " << v;
        }
        std::cout << std::endl;
    }
    /*
    std::string outputFile{"random3_target.fma"};
    std::cout << "Write targets in " << outputFile << std::endl;
    scalfmm::io::FFmaGenericWriter<double> writer_t(outputFile);
    writer_t.writeDataFromTree(tree_target, NbTargets);


    outputFile = "random3_source.fma";
    scalfmm::io::FFmaGenericWriter<double> writer_s(outputFile);
    writer_s.writeDataFromTree(tree_source, NbSources);
    */
    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::tree_height{},
                                                    args::block_size{}, args::order{}, args::thread_count{},
                                                    local_args::output_file(), local_args::check{}, local_args::ns{},
                                                    local_args::nt{}, local_args::scale{}, local_args::seed{});
    parser.parse(argc, argv);

    int seed{parser.get<local_args::seed>()};
    const int NbSources{parser.get<local_args::ns>()};
    const int NbTargets{parser.get<local_args::nt>()};
    std::cout << cpp_tools::colors::blue << "<params> seed:           " << seed << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> NbSources:      " << NbSources << cpp_tools::colors::reset
              << '\n';
    std::cout << cpp_tools::colors::blue << "<params> NbTargets:      " << NbTargets << cpp_tools::colors::reset
              << '\n';
    std::vector<value_type> scale(3, 1.0);
    scale = parser.get<local_args::scale>();

    const int NbLevels{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height :   " << NbLevels << cpp_tools::colors::reset << '\n';

    const int group_size{parser.get<args::block_size>()};
    const auto order{parser.get<args::order>()};

    std::cout << cpp_tools::colors::blue << "<params> Group Size :    " << group_size << cpp_tools::colors::reset
              << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    bool check_direct{parser.exists<local_args::check>()};
    const auto output_file{parser.get<local_args::output_file>()};
    const auto nb_threads{parser.get<args::thread_count>()};
    std::cout << cpp_tools::colors::blue << "<params> nb_threads :    " << nb_threads << cpp_tools::colors::reset
              << '\n';
#ifdef _OPENMP
    omp_set_dynamic(false);
    omp_set_num_threads(nb_threads);
#endif
    // if(!output_file.empty())
    // {
    //     std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
    //               << '\n';
    // }
    //

    fmm_run<dimension, value_type, fmm_operators_type>(NbLevels, group_size, order, seed, scale, NbSources, NbTargets,
                                                       check_direct, output_file);
}
