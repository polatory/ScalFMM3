// --------------------------------
// See LICENCE file at project root
// File : utils/generate.hpp
// --------------------------------

#ifndef SCALFMM_UTILS_GENERATE_HPP
#define SCALFMM_UTILS_GENERATE_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <random>
#include <utility>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/utils.hpp"

namespace scalfmm::utils
{
    template<typename ParticleType>
    auto generate_particles(std::size_t nb_particle, typename ParticleType::position_type center,
                            typename ParticleType::position_value_type width,const int seed=33)
      -> container::particle_container<ParticleType>
    {
        using particle_type = ParticleType;
        using value_type = typename ParticleType::position_value_type;
        using container_type = container::particle_container<ParticleType>;

        container_type container(nb_particle);
       // const auto seed{33};
        std::mt19937_64 gen(seed);
        constexpr value_type half{0.5};
        std::uniform_real_distribution<> dist(-width * half, width * half);
        auto it = std::begin(container);
        for(std::size_t i = 0; i < nb_particle; ++i)
        {
            meta::for_each(*it, [&dist, &gen](auto& v) { return v = dist(gen); });
            auto part = particle_type(*it);
            auto& pos = part.position();
            auto& out = part.outputs();
            pos += center;
            meta::repeat([](auto& e){ e = 0.; }, out);
            *it = part.as_tuple();
            ++it;
        }
        return container;
    }

    template<typename T, std::size_t dimension>
    constexpr auto get_center()
    {
        if constexpr(dimension == 1)
        {
            return container::point<T, dimension>{0.375};
        }
        if constexpr(dimension == 2)
        {
            return container::point<T, dimension>{0.375, 0.125};
        }
        if constexpr(dimension == 3)
        {
            return container::point<T, dimension>{0.375, 0.125, 0.125};
        }
    }

    template<typename ValueType, std::size_t dimension, std::size_t pv>
    struct get_particle_type
    {
    };

    // full direct for unit testing
    template<typename ValueType, std::size_t dimension, std::size_t physical_values, typename ContainerIterator,
             typename MatrixKernel, typename Outputs>
    inline auto full_direct_test(ContainerIterator begin, ContainerIterator end, MatrixKernel matrix_kernel,
                                 Outputs& out) -> void
    {
        using particle_type = typename utils::get_particle_type<ValueType, dimension, physical_values>::type;

        auto outputs_it = std::begin(out);
        for(auto it_p = begin; it_p < end; ++it_p)
        {
            auto pt_x = particle_type(*it_p).position();
            // 1 is the number of physical values (info coming from the MatrixKernel
            // get the potential
            for(auto it_p2 = begin; it_p2 < it_p; ++it_p2)
            {
                auto pt_y = particle_type(*it_p2).position();
                auto source = particle_type(*it_p2).attributes();
                meta::for_each(*outputs_it, *outputs_it, source,
                               [&matrix_kernel, &pt_x, &pt_y](auto const& o, auto const& s) {
                                   return o + (matrix_kernel.evaluate(pt_x, pt_y) * s);
                               });
            }
            for(auto it_p2 = it_p + 1; it_p2 < end; ++it_p2)
            {
                auto pt_y = particle_type(*it_p2).position();
                auto source = particle_type(*it_p2).attributes();
                meta::for_each(*outputs_it, *outputs_it, source,
                               [&matrix_kernel, &pt_x, &pt_y](auto const& o, auto const& s) {
                                   return o + (matrix_kernel.evaluate(pt_x, pt_y) * s);
                               });
            }
            ++outputs_it;
        }
    }

    template<typename T, typename U, typename F, std::size_t... Is>
    constexpr auto meta_compare_impl(T const& t, U const& u, F&& f, std::index_sequence<Is...> s) -> bool
    {
        return (std::invoke(std::forward<F>(f), meta::get<Is>(t), meta::get<Is>(u)) && ...);
    }

    template<typename T, typename U, typename F>
    constexpr auto meta_compare(T const& t, U const& u, F&& f = std::equal<>) -> bool
    {
        return meta_compare_impl(t, u, std::forward<F>(f), std::make_index_sequence<meta::tuple_size_v<T>>{});
    }

    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp)
    {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
               // unless the result is subnormal
               || std::fabs(x - y) < std::numeric_limits<T>::min();
    }
}   // namespace scalfmm::utils

#endif   // SCALFMM_UTILS_GENERATE_HPP
