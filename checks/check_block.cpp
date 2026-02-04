
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
#include "scalfmm/tree/leaf_view.hpp"
// #include "scalfmm/tree/leaf.hpp"
#include "scalfmm/utils/generate.hpp"

#include <iostream>
#include <random>
#include <vector>

template<typename OutputStream, typename ParticleType>
inline auto operator<<(OutputStream& os, std::vector<ParticleType>& particles) -> OutputStream&
{
    for(auto const& part: particles)
    {
        os << part << std::endl;
    }
    return os;
}

int main()
{
    using value_type = double;
    static constexpr std::size_t nb_inputs = 2;
    static constexpr std::size_t nb_outputs = 4;
    static constexpr std::size_t dimension = 3;
    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs, std::size_t>;

    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using block_type = typename leaf_type::storage_type;

    using container_type = std::vector<particle_type>;

    const std::size_t N{10};
    container_type particles(N);

    for(auto& part: particles)
    {
        int value = 0;
        scalfmm::meta::for_each(part.position(), [&value](auto& v) { v = ++value; });
        scalfmm::meta::for_each(part.inputs(), [&value](auto& v) { v = ++value; });
        scalfmm::meta::for_each(part.outputs(), [&value](auto& v) { v = ++value; });
        scalfmm::meta::for_each(part.variables(), [&value](auto& v) { v = ++value; });
    }

    std::cout << particles << std::endl;

    block_type block(particles, 2);

    std::cout << block << std::endl;

    leaf_type leaf1(std::make_pair(std::begin(block), std::begin(block) + N / 2), block.symbolics());
    leaf_type leaf2(std::make_pair(std::begin(block) + N / 2, std::end(block)), block.symbolics());

    auto print_leaf = [](auto const& leaf)
    {
        for(auto const p_ref: leaf)
        {
            const auto p = typename leaf_type::const_proxy_type(p_ref);
            std::cout << p << std::endl;
        }
    };

    std::cout << "\n - leaf 1: " << std::endl;
    print_leaf(leaf1);

    std::cout << "\n - leaf 2: " << std::endl;
    print_leaf(leaf2);

    return 0;
}
