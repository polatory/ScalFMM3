/**
 * @file check_reset.cpp
 * @author Antoine Gicquel
 * @brief This file checks the reset functions of various containers.
 *
 * @date 2025-05-01
 *
 */

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/is_valid.hpp"
#include "scalfmm/meta/type_pack.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/operators/l2p.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_of_views.hpp"
// #include "scalfmm/tree/leaf.hpp" // Old leaf format
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/generate.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

/**
 * @brief
 *
 * @tparam LeafType
 * @tparam ParticleType
 * @param container
 */
template<typename ParticleType, typename ContainerType>
auto init_leaf(scalfmm::component::leaf_view<ParticleType>& leaf, ContainerType const& container) -> void
{
    auto leaf_container_begin = leaf.particles().first;

    for(std::size_t index_part = 0; index_part < leaf.size(); ++index_part)
    {
        *leaf_container_begin = container.at(index_part).as_tuple();
        ++leaf_container_begin;
    }
}

/**
 * @brief
 *
 * @tparam ParticleType
 * @param container
 */
template<typename ParticleType>
auto init_container(std::vector<ParticleType>& container) -> void
{
    using particle_type = ParticleType;
    const std::size_t N{container.size()};

    for(std::size_t idx = 0; idx < N; ++idx)
    {
        int value = 0;
        particle_type& p = container[idx];
        scalfmm::meta::for_each(p.position(), [&value](auto& v) { return v = ++value; });
        scalfmm::meta::for_each(p.inputs(), [&value](auto& v) { return v = ++value; });
        scalfmm::meta::for_each(p.outputs(), [&value](auto& v) { return v = ++value; });
        scalfmm::meta::for_each(p.variables(), [&value](auto& v) { return v = ++value; });
    }
}

/**
 * @brief
 *
 * @tparam ParticleType
 * @param container
 */
template<typename ParticleType>
auto init_container(scalfmm::container::particle_container<ParticleType>& container) -> void
{
    const std::size_t N{container.size()};

    auto it = std::begin(container);
    for(std::size_t idx = 0; idx < N; ++idx)
    {
        int value = 0;
        scalfmm::meta::for_each(*it, [&value](auto& v) { return v = ++value; });
        ++it;
    }
}

/**
 * @brief
 *
 * @tparam OutputStreamType
 * @tparam ParticleType
 * @param os
 * @param particles
 */
template<typename OutputStreamType, typename ParticleType>
auto operator<<(OutputStreamType& os, const std::vector<ParticleType>& particles) -> OutputStreamType&
{
    for(auto const& part: particles)
    {
        os << part << std::endl;

        return os;
    }
}

/**
 * @brief
 *
 * @tparam ParticleType
 * @param N
 */
// template<typename ParticleType>
// auto run_old_leaf(std::size_t N) -> void
// {
//     using particle_type = ParticleType;
//     static constexpr std::size_t dimension = particle_type::dimension;

//     using value_type = typename particle_type::position_value_type;
//     using point_type = typename particle_type::position_type;
//     using leaf_type = scalfmm::component::leaf<particle_type>;
//     using container_type = scalfmm::container::particle_container<particle_type>;

//     // init base container
//     container_type leaf_container(N);
//     init_container(leaf_container);

//     // create leaf (old format) from container
//     const point_type center{scalfmm::utils::get_center<value_type, dimension>()};
//     const value_type width{0.25};
//     leaf_type leaf(std::cbegin(leaf_container), std::cend(leaf_container), N, center, width, 0);

//     std::cout << leaf.particles() << std::endl;

//     // reset positions in the leaf
//     leaf.reset_positions();
//     std::cout << leaf.particles() << std::endl;

//     // reset inputs in the leaf
//     leaf.reset_inputs();
//     std::cout << leaf.particles() << std::endl;

//     // reset outputs in the leaf
//     leaf.reset_outputs();
//     std::cout << leaf.particles() << std::endl;

//     // reset variables in the leaf
//     leaf.reset_variables();
//     std::cout << leaf.particles() << std::endl;

//     // create a new leaf again from container
//     leaf = leaf_type(std::cbegin(leaf_container), std::cend(leaf_container), N, center, width, 0);
//     std::cout << leaf.particles() << std::endl;

//     // rest all particles in the leaf
//     leaf.reset_particles();
//     std::cout << leaf.particles() << std::endl;
// }

/**
 * @brief
 *
 * @tparam ParticleType
 * @param N
 */
template<typename ParticleType>
auto run_leaf_view(std::size_t N) -> void
{
    using particle_type = ParticleType;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using container_type = std::vector<particle_type>;
    using storage_type = typename leaf_type::storage_type;

    // init container
    container_type container(N);
    init_container(container);

    // create block storage and leaf view on this storage
    storage_type storage(N, 1);
    leaf_type leaf(std::make_pair(std::begin(storage), std::begin(storage) + N), storage.symbolics());

    // init leaf (and the underlying storage) from the container
    init_leaf(leaf, container);
    std::cout << storage << std::endl;

    // create a leaf view only on the first half of the storage
    leaf_type first_leaf(std::make_pair(std::begin(storage), std::begin(storage) + N / 2), storage.symbolics());

    // reset the positions on the first half of the storage
    first_leaf.reset_positions();
    std::cout << storage << std::endl;

    // reset the inputs on the first half of the storage
    first_leaf.reset_inputs();
    std::cout << storage << std::endl;

    // reset the outputs on the first half of the storage
    first_leaf.reset_outputs();
    std::cout << storage << std::endl;

    // reset the variables on the first half of the storage
    first_leaf.reset_variables();
    std::cout << storage << std::endl;

    // init leaf (and the underlying storage) from the container
    init_leaf(leaf, container);
    std::cout << storage << std::endl;

    // reset the particles on the first half of the storage
    first_leaf.reset_particles();
    std::cout << storage << std::endl;
}

template<typename ParticleType>
auto run_variadic_container(std::size_t N) -> void
{
    using particle_type = ParticleType;
    using container_type = scalfmm::container::particle_container<particle_type>;

    // init container
    container_type container(N);
    init_container(container);
    std::cout << container << std::endl;

    // reset positions in the container
    container.reset_positions();
    std::cout << container << std::endl;

    // reset the inputs in the container
    container.reset_inputs();
    std::cout << container << std::endl;

    // reset the outputs in the container
    container.reset_outputs();
    std::cout << container << std::endl;

    // reset the variables in the container
    container.reset_variables();
    std::cout << container << std::endl;

    // init again the container
    init_container(container);
    std::cout << container << std::endl;

    // reset the particles in the container
    container.reset_particles();
    std::cout << container << std::endl;
}

auto main() -> int
{
    using value_type = double;
    static constexpr std::size_t dimension = 3;
    static constexpr std::size_t nb_inputs = 4;
    static constexpr std::size_t nb_outputs = 4;

    std::size_t N{10};

    using particle_type = scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type,
                                                       nb_outputs, std::size_t, float, double, int>;

    // run_old_leaf<particle_type>(N);
    run_leaf_view<particle_type>(N);
    run_variadic_container<particle_type>(N);

    return 0;
}
