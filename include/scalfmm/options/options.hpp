// --------------------------------
// See LICENCE file at project root
// File : options/options.hpp
// --------------------------------
#ifndef SCALFMM_OPTIONS_OPTIONS_HPP
#define SCALFMM_OPTIONS_OPTIONS_HPP

#include <sstream>
#include <string>
#include <string_view>
#include <tuple>

#include "scalfmm/meta/utils.hpp"

namespace scalfmm::options
{
    template<typename D>
    struct setting
    {
        using type = setting<D>;
        using derived_type = D;

        constexpr auto value() const noexcept -> std::string_view { return static_cast<derived_type>(this)->value(); }
    };

    template<typename... S>
    struct settings : S...
    {
        using type = settings<S...>;

        template<typename... Setting>
        constexpr settings(Setting... s)
        {
        }
    };

    template<typename... S>
    settings(S... s) -> settings<typename S::type...>;

    struct dense_ : setting<dense_>
    {
        using type = dense_;
        static constexpr auto value() noexcept -> std::string_view { return "dense"; };
    };

    struct fft_ : setting<fft_>
    {
        using type = fft_;
        static constexpr auto value() noexcept -> std::string_view { return "fft"; };
    };

    struct low_rank_ : setting<low_rank_>
    {
        using type = low_rank_;
        static constexpr auto value() noexcept -> std::string_view { return "low_rank"; };
    };

    template<typename... S>
    struct seq_
      : setting<seq_<S...>>
      , S...
    {
        using type = seq_<S...>;
        using inner_settings = settings<S...>;
        static constexpr auto value() noexcept -> std::string_view { return "seq"; };
    };

    template<typename... S>
    struct omp_
      : setting<omp_<S...>>
      , S...
    {
        using type = omp_<S...>;
        using inner_settings = settings<S...>;
        static constexpr auto value() noexcept -> std::string_view { return "omp"; };
    };

    struct timit_ : setting<timit_>
    {
        using type = timit_;
        static constexpr auto value() noexcept -> std::string_view { return "timit"; };
    };

    template<typename S = fft_>
    struct uniform_
        : setting<uniform_<S>>
        , S
    {
        using type = uniform_<S>;
        static constexpr auto value() noexcept -> std::string_view { return "uniform_"; };
        // {
        //     std::stringstream ss;
        //     ss << "uniform_" << S().value();
        //     return std::string(ss.str());
        // };
    };

    template<typename S = low_rank_>
    struct chebyshev_
        : setting<chebyshev_<S>>
        , S
    {
        using type = chebyshev_<S>;
        static constexpr auto value() noexcept -> std::string_view { return "chebyshev_"; };

        // static auto value() noexcept -> std::string
        // {
        //     std::stringstream ss;
        //     ss << "chebyshev_" << S().value();
        //     return std::string(ss.str());
        // };
    };


    static constexpr auto uniform_dense = uniform_<dense_>{};
    static constexpr auto uniform_fft = uniform_<fft_>{};
    static constexpr auto uniform_low_rank = uniform_<low_rank_>{};
    static constexpr auto chebyshev_dense = chebyshev_<dense_>{};
    static constexpr auto chebyshev_low_rank = chebyshev_<low_rank_>{};
    static constexpr auto dense = dense_{};
    static constexpr auto fft = fft_{};
    static constexpr auto low_rank = low_rank_{};
    static constexpr auto omp = omp_{};
    static constexpr auto omp_timit = omp_<timit_>{};
    static constexpr auto seq = seq_{};
    static constexpr auto seq_timit = seq_<timit_>{};
    static constexpr auto timit = timit_{};

    /**
     * @brief Check if S2 is inside S1
     *
     * Here we check if option op used for the algorithm contains omp.
     *  In other word, we check if the algorithm is the OpenMP one
     * \code {.c++}
     * has(scalfmm::options::_s(op),scalfmm::options::_s(scalfmm::options::omp))
     * \endcode
     *
     * @tparam S1
     * @tparam S2
     * @param s1 the first setting
     * @param s2 the second one
     * @return true
     * @return false
     */
    template<typename... S1, typename... S2>
    static constexpr auto has(settings<S1...> s1, settings<S2...> s2) -> bool
    {
        return (... || std::is_base_of_v<S2, decltype(s1)>);
    }

    template<typename... S1, typename... S>
    static constexpr auto has(settings<S1...> s1, S... s) -> bool
    {
        return (... || std::is_base_of_v<S, decltype(s1)>);
    }

    template<typename... S1, typename... S2>
    static constexpr auto match(settings<S1...> s1, settings<S2...> s2) -> bool
    {
        return (sizeof...(S1) == sizeof...(S2)) && (... && std::is_base_of_v<S2, decltype(s1)>);
    }

    template<typename... S1, typename... S>
    static constexpr auto match(settings<S1...> s1, S... s) -> bool
    {
        return ((sizeof...(S1) == sizeof...(S)) && (... && std::is_base_of_v<S, decltype(s1)>));
    }

    template<typename... S1, typename... S2>
    static constexpr auto support(settings<S1...> s1, settings<S2...> s2) -> bool
    {
        return (... && std::is_base_of_v<S1, decltype(s2)>);
    }

    template<typename... S>
    static constexpr auto is_settings(settings<S...> s) -> bool
    {
        return true;
    }

    template<typename T>
    static constexpr auto is_settings(T t) -> bool
    {
        return false;
    }

    template<typename... S>
    static constexpr auto _s(S... s)
    {
        return settings<S...>{};
    }

    template<typename... S>
    static constexpr auto _s(settings<S...> s)
    {
        return s;
    }
}   // namespace scalfmm::options


#define DECLARE_OPTIONED_CALLEE(NAME)                                                    \
struct NAME_                                                                             \
{                                                                                        \
    template<typename Arg, typename... Args>                                             \
    inline static constexpr auto call(Arg&& arg, Args&&... args) noexcept                \
    {                                                                                    \
        return impl::NAME(std::forward<Arg>(arg), std::forward<Args>(args)...);          \
    }                                                                                    \
    template<typename... S>                                                              \
    inline auto constexpr operator[](scalfmm::options::settings<S...> s)  const noexcept \
    {                                                                                    \
        return [s](auto&&... args)                                                       \
        { return NAME_::call(s, std::forward<decltype(args)>(args)...); };               \
    }                                                                                    \
    template<typename... Args>                                                           \
    inline auto constexpr operator()(Args&&... args)  const noexcept                     \
    {                                                                                    \
        return NAME_::call(std::forward<Args>(args)...);                                 \
    }                                                                                    \
};                                                                                       \
inline const NAME_ NAME = {};                                                            \


#endif // SCALFMM_OPTIONS_OPTIONS_HPP
