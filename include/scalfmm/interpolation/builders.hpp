// --------------------------------
// See LICENCE file at project root
// File : interpolation/builders.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_BUILDERS_HPP
#define SCALFMM_INTERPOLATION_BUILDERS_HPP

#include <scalfmm/meta/const_functions.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>
#include <cstddef>
#include <tuple>
#include <utility>

#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xcontainer.hpp"
#include "xtensor/xtensor_forward.hpp"

namespace scalfmm::interpolation
{
    template<std::size_t, typename T>
    constexpr auto id(T t)
    {
        return std::forward<T>(t);
    }

    template<typename Gen, std::size_t... I>
    constexpr auto get_generator(Gen&& gen, std::index_sequence<I...> /*unused*/)
    {
        return std::move(xt::stack(std::make_tuple(id<I>(gen)...)));
    }

    template<std::size_t dim>
    constexpr auto get(const std::size_t order)
    {
        return get_generator(xt::linspace(std::size_t{0}, order - 1, order), std::make_index_sequence<dim>{});
    }

    // TODO for nodes
    // auto lin = xt::stack(xt::xtuple(xt::linspace(std::size_t{0},order-1,order)));
    // xt::xarray<std::size_t> saved;
    // for(std::size_t i=0; i<dimension; ++i)
    //{
    //  auto r = xt::stack(xt::xtuple(xt::ones<std::size_t>({meta::pow(order,(dimension-1-i))})));
    //  auto tmp = xt::linalg::kron(lin,r);
    //  auto l = xt::stack(xt::xtuple(xt::ones<std::size_t>({meta::pow(order,i)})));
    //  auto reshaped = xt::linalg::kron(l,tmp);
    //  reshaped.reshape({1,-1});
    //  auto res = xt::stack(xt::xtuple(reshaped,saved));
    //  saved = res;
    //}
    //  std::cout << "ids: " << saved << '\n';

    template<std::size_t dim>
    struct nodes
    {
        static_assert(dim < 4, "Dimension for interpolation node not supported.");
    };

    template<>
    struct nodes<1>
    {
        inline static auto get(std::size_t order) { return xt::linspace(std::size_t{0}, order - 1, order); }
    };

    template<>
    struct nodes<2>
    {
        inline static auto get(std::size_t order)
        {
            const auto s{meta::pow(order, 2)};
            const xt::xarray<std::size_t>::shape_type shape{2, s};
            xt::xarray<std::size_t> ns(shape);
            for(std::size_t i = 0; i < s; ++i)
            {
                ns(0, i) = i % order;
                ns(1, i) = (i / order) % order;
            }
            return ns;
        }
    };

    template<>
    struct nodes<3>
    {
        inline static auto get(std::size_t order)
        {
            const auto s{meta::pow(order, 3)};
            const xt::xarray<std::size_t>::shape_type shape{3, s};
            xt::xarray<std::size_t> ns(shape);
            for(std::size_t i = 0; i < s; ++i)
            {
                ns(0, i) = i % order;
                ns(1, i) = (i / order) % order;
                ns(2, i) = i / (order * order);
            }
            return ns;
        }
    };

    template<>
    struct nodes<4>
    {
        inline static auto get(std::size_t order)
        {
            const auto s{meta::pow(order, 4)};
            const xt::xarray<std::size_t>::shape_type shape{4, s};
            xt::xarray<std::size_t> ns(shape);
            for(std::size_t i = 0; i < s; ++i)
            {
                ns(0, i) = i % order;
                ns(1, i) = (i / (order)) % order;
                ns(2, i) = (i / (order * order)) % order;
                ns(3, i) = i / (order * order * order);
            }
            return ns;
        }
    };

    // Builders for relative center

}   // namespace scalfmm::interpolation

#endif   // SCALFMM_INTERPOLATION_BUILDERS_HPP
