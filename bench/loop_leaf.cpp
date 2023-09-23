// @FUSE_OMP

/**
 * \brief Check the access on the particles inside the leaves
 *\code
 * bench/Release/loop_leaf-th 3 -gs 10
 *\endcode
 */
// #define SAME_MK
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <valarray>

#include "scalfmm/container/access.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
//
//
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"

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
#include "scalfmm/tools/fma_loader.hpp"

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
    // struct output_file
    // {
    //     cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
    //     const char* description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
    //     using type = std::string;
    // };
    struct ns
    {
        cpp_tools::cl_parser::str_vec flags = {"--number_particles", "-n"};
        const char* description = "Number of particles.";
        using type = int;
        type def = 2000000;
    };
    struct seed
    {
        cpp_tools::cl_parser::str_vec flags = {"-seed"};
        const char* description = "Seed to generate random particles.";
        using type = int;
        type def = 1323;
    };
}   // namespace local_args

using timer_type = cpp_tools::timers::timer<std::chrono::milliseconds>;
using timer_acc_type = cpp_tools::timers::timer<std::chrono::microseconds>;

using value_type = double;
constexpr int dimension = 3;
//
using options = scalfmm::options::chebyshev_<scalfmm::options::low_rank_>;
// using options = scalfmm::options::uniform_<scalfmm::options::fft_>;
// far field
//  Uniform interpolation and fft optimization

using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
using interpolation_type = scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type, options>;
//
static constexpr std::size_t nb_inputs{1};
static constexpr std::size_t nb_outputs{1 + dimension};
constexpr int nb_input{nb_inputs};   // should be zero

using point_type = scalfmm::container::point<value_type, dimension>;

using particle_type =
  scalfmm::container::particle<value_type, dimension, value_type, nb_input, value_type, nb_outputs, std::size_t>;

using leaf_type = scalfmm::component::leaf_view<particle_type>;
// this struct is crucial for having the good type for iterator and group types (source) in
// the target leaves and groups

template<typename Tree, typename Value_type, typename Vect>
auto sum_output(Tree const& tree, const Value_type& Q, const int iter, Vect& res) -> void
{
    Value_type potential = Value_type(0.);
    Value_type fx = Value_type(0.), fy = Value_type(0.), fz = Value_type(0.);
    //
    timer_acc_type time{};
    time.tic();
    for(auto pg: tree.group_of_leaves())
    {
        //  std::cout << pg->storage() << "\n";
        auto nb_particles = pg->storage().size();
        auto ptr_potential = pg->storage().ptr_on_output();
        auto ptr_pos_x = reinterpret_cast<Value_type*>(pg->storage().particles());
        auto ptr_pos_y = ptr_pos_x + nb_particles;
        auto ptr_fx = ptr_potential + nb_particles;
        auto ptr_fy = ptr_fx + nb_particles;
        auto ptr_fz = ptr_fy;
        auto ptr_pos_z = ptr_pos_y;
        if constexpr(dimension == 3)
        {
            ptr_pos_z = ptr_pos_y + nb_particles;
            ptr_fz = ptr_fy + nb_particles;
        }
        auto ptr_var0 = reinterpret_cast<std::size_t*>(ptr_fz + nb_particles);

        // std::cout << "First output: " << ptr_potential << " " << ptr_fx << " " << ptr_fy << " " << ptr_fz << " \n ";
        for(auto i = 0; i < nb_particles; ++i)
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
    std::array<value_type, Tree::leaf_type::particle_type::outputs_size> acc{};
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
//
template<int dimension, typename value_type>
auto run_loop(const int& tree_height, const int& group_size, int const& order, int& seed, const int& Nb_particles)
  -> int
{
    bool display_container = false;
    bool display_tree = false;

    timer_type time{};

    constexpr int zeros{nb_inputs};   // should be zero
                                      // Construct the container of particles
    using container_type = scalfmm::container::particle_container<particle_type>;
    using box_type = scalfmm::component::box<point_type>;
    //
    using cell_type = scalfmm::component::cell<typename interpolation_type::storage_type>;
    using tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "Working on particles ..." << std::endl;
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<value_type> dist(-1, 1);
    seed48(reinterpret_cast<short unsigned int*>(&seed));

    std::cout << cpp_tools::colors::green << "Creating & Inserting ...\n" << cpp_tools::colors::reset;
    //
    seed48(reinterpret_cast<short unsigned int*>(&seed));

    time.tic();
    point_type box_center(0.);

    value_type box_width{2.0};
    container_type container(Nb_particles);
    box_type box(box_width, box_center);

    particle_type p;
    for(int i = 0; i < Nb_particles; ++i)
    {   // cibles

        scalfmm::meta::repeat([&gen, &dist](auto& v) { v = dist(gen); }, p.position());

        p.inputs(0) = std::abs(dist(gen));
        // set the outputs

        p.variables(i);
        std::cout << "target " << i << " " << p << std::endl;
        container.insert_particle(i, p);
    }

    // build trees
    bool sorted = false;
    time.tic();

    tree_type tree(tree_height, order, box, group_size, group_size, container, sorted);

    time.tac();
    std::cout << "Done  "
              << "(@Building trees = " << time.elapsed() << " ms)." << std::endl;

    std::cout << "bounding_box " << box << std::endl;
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;

    /////////////////////////////////////////////////////////////////////////////////////
    //
    //              Compute source-target interaction though FMM
    //
    /////////////////////////////////////////////////////////////////////////////////////
    box_width = box.width(0);
    // Far field
    far_matrix_kernel_type mk_far{};
    interpolation_type interpolator(mk_far, order, tree_height, box_width);
    //
    value_type Q{1.0};
    std::valarray<value_type> res(0.0, nb_outputs), error(0.0, nb_outputs);
    for(int iter = 0; iter < 1; ++iter)
    {
        std::cout << "iter = " << iter << "  Q " << Q << std::endl;
        scalfmm::component::for_each_leaf(
          std::begin(tree), std::end(tree),
          [&dist, &gen](auto& leaf)
          {
              //   auto part_it = std::begin(leaf.particles());
              auto begin = scalfmm::container::begin(leaf.particles());
              auto end = scalfmm::container::end(leaf.particles());
              auto outputs_begin = scalfmm::container::outputs_begin(leaf.particles());
              auto outputs_end = scalfmm::container::outputs_end(leaf.particles());
              auto it_part = begin;
              // generate random values for all the outputs in the leaves
              for(auto it = outputs_begin; it != outputs_end; ++it, ++it_part)
              {
                  scalfmm::meta::repeat([&gen, &dist](auto& v) { v = dist(gen); }, *it);
              }
              // a lazy light iterator on the inputs of particle inside the container
              auto inputs_begin_lazy = scalfmm::container::inputs_begin(leaf.particles());
              // we dereference to evaluate the lazy pointer
              auto inputs_begin = *inputs_begin_lazy;
              //  You get the first input value and you take
              // its address in order to increment if
              auto q = &std::get<0>(inputs_begin);
              // construct a sequence to access directly to the force in the output
              using range_force = scalfmm::meta::make_range_sequence<1, particle_type::outputs_size>;
              // the outputs are [ p, fx,fy, fz] and we construct [ p, q*fx,q*fy, q*fz]
              // where q is the first input of teh particle
              for(auto it = outputs_begin; it != outputs_end; ++it, ++q)
              {
                  // out =[ p, fx,fy, fz]
                  scalfmm::meta::repeat([q](auto& v) { v *= *q; }, scalfmm::meta::sub_tuple(*it, range_force{}));
              }
          });

        sum_output_2(tree, Q, iter, res);
        sum_output_1(tree, Q, iter, res);
        sum_output(tree, Q, iter, res);
    }

    return 0;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    // Parameter handling
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::tree_height{}, args::block_size{},
                                        args::thread_count{}, local_args::ns{}, local_args::seed{});
    parser.parse(argc, argv);

    int seed{parser.get<local_args::seed>()};
    const int Nb_particles{parser.get<local_args::ns>()};
    std::cout << cpp_tools::colors::blue << "<params> seed:           " << seed << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Nb_particles:   " << Nb_particles << cpp_tools::colors::reset
              << '\n';

    const int NbLevels{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height :   " << NbLevels << cpp_tools::colors::reset << '\n';

    const int group_size{parser.get<args::block_size>()};
    std::size_t order = 3;   // not used here

    std::cout << cpp_tools::colors::blue << "<params> Group Size :    " << group_size << cpp_tools::colors::reset
              << '\n';
    std::cout << cpp_tools::colors::blue << "<params> Order :         " << order << cpp_tools::colors::reset << '\n';

    const auto nb_threads{parser.get<args::thread_count>()};
    std::cout << cpp_tools::colors::blue << "<params> nb_threads :    " << nb_threads << cpp_tools::colors::reset
              << '\n';
#ifdef _OPENMP
    omp_set_dynamic(false);
    omp_set_nested(false);
    omp_set_num_threads(nb_threads);
#endif

    run_loop<dimension, value_type>(NbLevels, group_size, order, seed, Nb_particles);
}
