
// Be carefull with this header file
#include "scalfmm/container/access.hpp"

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/p2p.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
// #include "scalfmm/tree/leaf.hpp"
#include "scalfmm/tree/group_of_views.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/generate.hpp"
#include "scalfmm/utils/parameters.hpp"

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include <cassert>
#include <chrono>

namespace local_args
{
    struct nbParticles : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--number_particles", "-n", "-N"};
        std::string description = "Number of particles to generate";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
    };
    struct dimensionSpace : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--dim", "-d"};
        std::string description = "Dimension of the space (1,2 or 3)";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
    };
}   // namespace local_args

template<typename LeafType, typename ContainerType>
auto init_leaf(LeafType& leaf, ContainerType const& container) -> void
{
    auto leaf_container_begin = leaf.particles().first;

    for(std::size_t index_part = 0; index_part < leaf.size(); ++index_part)
    {
        *leaf_container_begin = container.at(index_part).as_tuple();
        ++leaf_container_begin;
    }
}

using timer_type = cpp_tools::timers::timer<std::chrono::microseconds>;

template<int Dimension, typename value_type, typename matrix_kernel_type>
auto run_inner(const int nb_exp, const std::size_t nb_particles, matrix_kernel_type& near) -> void
{
    static constexpr std::size_t nb_inputs_near{matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{matrix_kernel_type::kn};
    // ---------------------------------------
    using particle_type = scalfmm::container::particle<value_type, Dimension, value_type, nb_inputs_near, value_type,
                                                       nb_outputs_near, value_type>;
    using point_type = scalfmm::container::point<value_type, Dimension>;
    // using leaf_type = scalfmm::component::leaf<particle_type>;

    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using storage_type = typename leaf_type::storage_type;

    point_type center{scalfmm::utils::get_center<value_type, Dimension>()};

    const value_type width{0.125};

    auto target_container{scalfmm::utils::generate_particles<particle_type>(nb_particles, center, width)};
    //std::cout << container << std::endl;
    // leaf_type target_leaf(std::cbegin(container), std::cend(container), container.size(), center, width, 0);

    // Create a target leaf view (view on a distinct storage)
    storage_type target_storage(nb_particles, 1);
    // leaf_type target_leaf(std::make_pair(std::begin(target_storage), std::end(target_storage)),
    //                       target_storage.symbolics());
    leaf_type target_leaf(std::make_pair(std::begin(target_storage), std::begin(target_storage) + nb_particles),
                          target_storage.symbolics());
    target_leaf.center() = center;
    target_leaf.width() = width;
    init_leaf(target_leaf, target_container);

    // Shift center of the first leaf
    point_type shift(0.25);
    center += shift;

    // shifted container
    std::size_t nb_source_particles = nb_particles / 2;
    auto source_container{scalfmm::utils::generate_particles<particle_type>(nb_source_particles, center, width)};
    // leaf_type source_leaf(std::cbegin(container), std::cend(container), container.size(), center, width, 0);

    // Create a source leaf view (view on a distinct storage)
    storage_type source_storage(nb_particles / 2, 1);
    // leaf_type source_leaf(std::make_pair(std::begin(source_storage), std::end(source_storage)),
    //                       source_storage.symbolics());
    leaf_type source_leaf(std::make_pair(std::begin(source_storage), std::begin(source_storage) + nb_source_particles),
                          source_storage.symbolics());
    source_leaf.center() = center;
    source_leaf.width() = width;
    init_leaf(source_leaf, source_container);

    timer_type timer{};
    timer.tic();
    for(int i = 0; i < nb_exp; ++i)
    {
        scalfmm::operators::p2p_inner(near, target_leaf);
    }
    timer.tac();
    std::cout << "time for p2p_inner " << timer.elapsed() << std::endl;

    const std::array<bool, 3> pbc{false, false, false};
    std::array<leaf_type*, 1> neighbors{&source_leaf};
    timer.tic();
    for(int i = 0; i < nb_exp; ++i)
    {
        scalfmm::operators::p2p_full_mutual(near, target_leaf, neighbors, 1, pbc, width);
    }
    timer.tac();
    auto t1 = timer.elapsed();
    std::cout << "time for p2p_full_mutual " << t1 << std::endl;

    timer.tic();
    for(int i = 0; i < nb_exp; ++i)
    {
        scalfmm::operators::p2p_outer(near, target_leaf, source_leaf, shift);
        scalfmm::operators::p2p_outer(near, source_leaf, target_leaf, shift);
    }
    timer.tac();
    auto t2 = timer.elapsed();
    std::cout << "time for 2 p2p_outer = p2p_full_mutual " << t2 << " gain: " << t2 - t1 << " % "
              << 1 - double(t2) / double(t1) << std::endl;

    timer.tic();
    for(int i = 0; i < nb_exp; ++i)
    {
        scalfmm::operators::p2p_outer(near, target_leaf, source_leaf, shift);
    }
    timer.tac();
    std::cout << "time for p2p_outer " << timer.elapsed() << std::endl;

    //
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling

    auto parser = cpp_tools::cl_parser::make_parser(local_args::nbParticles{}, local_args::dimensionSpace{},
                                                    cpp_tools::cl_parser::help{});

    parser.parse(argc, argv);

    std::cout << cpp_tools::colors::blue << "Entering P2P test...\n" << cpp_tools::colors::reset;
    const int nb_particles{parser.get<local_args::nbParticles>()};
    std::cout << cpp_tools::colors::blue << "<params> nbParticles : " << nb_particles << cpp_tools::colors::reset
              << '\n';
    const int dimension{parser.get<local_args::dimensionSpace>()};
    std::cout << cpp_tools::colors::blue << "<params> dim : " << dimension << cpp_tools::colors::reset << '\n';

    int nb_experiments = 50;

    if(dimension == 3)
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<3>;
        matrix_kernel_type mk{};
        run_inner<3, double>(nb_experiments, nb_particles, mk);
    }
    else
    {
        throw std::invalid_argument("The dimension is wrong (only dimension = 3)");
    }
}
