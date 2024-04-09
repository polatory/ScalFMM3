// --------------------------------
// See LICENCE file at project root
// File : simd/memeory.hpp
// --------------------------------
#ifndef SCALFMM_SIMD_MEMORY_HPP
#define SCALFMM_SIMD_MEMORY_HPP

#include <xsimd/xsimd.hpp>
#include <scalfmm/meta/traits.hpp>
#include <scalfmm/meta/utils.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <cstddef>
#include <functional>


namespace scalfmm::simd
{
    struct aligned
    {
    };
    struct unaligned
    {
    };
    struct splated
    {
    };

    namespace impl
    {
        template<typename SimdType, typename Tuple, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_splat_value(Tuple const& values_to_splat, std::index_sequence<Is...> s)
        {
            return std::make_tuple(SimdType(meta::get<Is>(values_to_splat))...);
        }

        template<typename SimdType, typename TupleOfIt, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_splat_value_from_it(TupleOfIt const& values_to_splat, std::index_sequence<Is...> s)
        {
            return std::make_tuple(SimdType(*meta::get<Is>(values_to_splat))...);
        }

        template<typename Position, typename TuplePtr, typename Aligned, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_position(TuplePtr const& particle_ptrs, Aligned a,
                                                          std::index_sequence<Is...> s)
        {
            using value_type = typename Position::value_type;
            if constexpr(meta::is_simd<value_type>::value)
            {
                if constexpr(std::is_same_v<Aligned, aligned>)
                {
                    return Position{xsimd::load_aligned(&meta::get<Is>(*particle_ptrs))...};
                }
                else if constexpr(std::is_same_v<Aligned, unaligned>)
                {
                    return Position{xsimd::load_unaligned(&meta::get<Is>(*particle_ptrs))...};
                }
                else if constexpr(std::is_same_v<Aligned, splated>)
                {
                    return Position{value_type(meta::get<Is>(*particle_ptrs))...};
                }
            }
            else
            {
                return Position{meta::get<Is>(*particle_ptrs)...};
            }
        }

        template<typename StoredType, typename Tuple, typename TuplePtr, typename Aligned, std::size_t... Is>
        constexpr inline void store_tuple(TuplePtr& variadic_adaptor_iterator, Tuple const& src, Aligned a,
                                          std::index_sequence<Is...> s)
        {
            if constexpr(meta::is_simd<StoredType>::value)
            {
                if constexpr(std::is_same_v<Aligned, aligned>)
                {
                    meta::noop_f{
                      (xsimd::store_aligned(&meta::get<Is>(*variadic_adaptor_iterator), meta::get<Is>(src)), 0)...};
                }
                else if constexpr(std::is_same_v<Aligned, unaligned>)
                {
                    meta::noop_f{
                      (xsimd::store_unaligned(&meta::get<Is>(*variadic_adaptor_iterator), meta::get<Is>(src)), 0)...};
                }
            }
            else
            {
                meta::noop_f{(meta::get<Is>(*variadic_adaptor_iterator) = meta::get<Is>(src), 0)...};
            }
        }

        template<typename LoadedType, typename TuplePtr, typename Aligned, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_tuple(TuplePtr const& particle_ptrs, Aligned a,
                                                       std::index_sequence<Is...> s)
        {
            if constexpr(meta::is_simd<LoadedType>::value)
            {
                if constexpr(std::is_same_v<Aligned, aligned>)
                {
                    return std::make_tuple(xsimd::load_aligned(&meta::get<Is>(*particle_ptrs))...);
                }
                else if constexpr(std::is_same_v<Aligned, unaligned>)
                {
                    return std::make_tuple(xsimd::load_unaligned(&meta::get<Is>(*particle_ptrs))...);
                }
                else if constexpr(std::is_same_v<Aligned, splated>)
                {
                    return std::make_tuple(LoadedType(meta::get<Is>(*particle_ptrs))...);
                }
            }
            else
            {
                return std::make_tuple(meta::get<Is>(*particle_ptrs)...);
            }
        }

        //template<typename T, typename Type, typename TT, typename Container, typename... Types, std::size_t... Is>
        //[[nodiscard]] constexpr inline auto apply_f(std::index_sequence<Is...> s, Type T::*f, TT&& tt,
        //                                            Container&& input, Types&&... ts)
        //{
        //    using return_type = typename std::decay_t<Container>;
        //    return return_type{
        //      {std::invoke(f, tt, meta::get<Is>(std::forward<Container>(input)), std::forward<Types>(ts)...)...}};
        //}

        template<typename F, typename Container, typename... Types, std::size_t... Is>
        [[nodiscard]] constexpr inline auto apply_f(std::index_sequence<Is...> s, F&& f, Container&& input,
                                                    Types&&... ts)
        {
            using return_type = typename std::decay_t<Container>;
            return return_type{{std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<Container>(input)),
                                            std::forward<Types>(ts)...)...}};
        }

    }   // namespace impl

    template<typename SimdType, typename Tuple>
    [[nodiscard]] constexpr inline auto load_splat_value(Tuple const& values_to_splat)
    {
        return impl::load_splat_value<SimdType>(values_to_splat, std::make_index_sequence<meta::tuple_size_v<Tuple>>{});
    }

    template<typename SimdType, typename TupleOfIt>
    [[nodiscard]] constexpr inline auto load_splat_value_from_it(TupleOfIt const& values_to_splat)
    {
        return impl::load_splat_value_from_it<SimdType>(values_to_splat, std::make_index_sequence<meta::tuple_size_v<TupleOfIt>>{});
    }

    template<typename Position, typename TuplePtr, typename Aligned = aligned>
    [[nodiscard]] constexpr inline auto load_position(TuplePtr const& particle_ptrs, Aligned a = Aligned{})
    {
        return impl::load_position<Position>(particle_ptrs, a, std::make_index_sequence<Position::dimension>{});
    }

    template<typename LoadedType, typename TuplePtr, typename Aligned = aligned>
    [[nodiscard]] constexpr inline auto load_tuple(TuplePtr const& variadic_adaptor_iterator, Aligned a = Aligned{})
    {
        return impl::load_tuple<LoadedType>(
          variadic_adaptor_iterator, a,
          std::make_index_sequence<meta::tuple_size_v<decltype(std::declval<TuplePtr>().operator*())>>{});
    }

    // template<typename Particle, typename LoadedType, typename TuplePtr, typename Aligned = aligned>
    //[[nodiscard]] constexpr inline auto load_attributes(TuplePtr const& particle_ptrs, Aligned a = Aligned{})
    //{
    //    return impl::load_attributes<LoadedType>(
    //      particle_ptrs, a,
    //      meta::add_to_sequence<Particle::dimension>(std::make_index_sequence<Particle::attributes_size>{}));
    //}

    // template<typename Position, typename TuplePtr, typename Aligned = aligned>
    //[[nodiscard]] constexpr inline auto load_forces(TuplePtr const& particle_ptrs, Aligned a = Aligned{})
    //{
    //    return impl::load_position<Position>(particle_ptrs, a, std::make_index_sequence<Position::dimension>{});
    //}

    template<typename Position, typename TuplePtr, typename Aligned = aligned>
    constexpr inline void store_position(TuplePtr& particle_ptrs, Position const& src, Aligned a = Aligned{})
    {
        impl::store_tuple<typename Position::value_type>(particle_ptrs, src, a,
                                                         std::make_index_sequence<Position::dimension>{});
    }

    template<typename StoredType, typename TupleSrc, typename TuplePtr, typename Aligned = aligned>
    constexpr inline void store_tuple(TuplePtr& variadic_adaptor_iterator, TupleSrc const& src, Aligned a = Aligned{})
    {
        impl::store_tuple<StoredType>(variadic_adaptor_iterator, src, a,
                                      std::make_index_sequence<meta::tuple_size_v<TupleSrc>>{});
    }

    // template<typename StoredType, typename TupleAttributes, typename TuplePtr, typename Aligned = aligned>
    // constexpr inline void store_attributes(TuplePtr& attributes_ptrs, TupleAttributes const& src, Aligned a =
    // Aligned{})
    //{
    //    impl::store_tuple<StoredType>(attributes_ptrs, src, a,
    //                                  std::make_index_sequence<meta::tuple_size<TupleAttributes>::value>{});
    //}

    template<std::size_t Size, typename F, typename Container, typename... Types>
    [[nodiscard]] constexpr inline auto apply_f(F&& f, Container&& input, Types&&... ts)
    {
        return impl::apply_f(std::make_index_sequence<Size>{}, std::forward<F>(f), std::forward<Container>(input),
                             std::forward<Types>(ts)...);
    }

    //template<std::size_t Size, typename F, typename TT, typename Container, typename... Types>
    //[[nodiscard]] constexpr inline auto apply_f(F&& f, TT& tt, Container&& input, Types&&... ts)
    //{
    //    return impl::apply_f(std::make_index_sequence<Size>{}, std::forward<F>(f), std::forward<TT>(tt),
    //                         std::forward<Container>(input), std::forward<Types>(ts)...);
    //}

}   // namespace scalfmm::simd

#endif   // SCALFMM_SIMD_MEMORY_HPP
