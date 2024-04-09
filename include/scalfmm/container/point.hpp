// See LICENCE file at project root
//
#ifndef SCALFMM_CONTAINER_POINT_HPP
#define SCALFMM_CONTAINER_POINT_HPP

#include <scalfmm/container/reference_sequence.hpp>
#include <scalfmm/meta/utils.hpp>
#include <array>
#include <initializer_list>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <functional>
#include <cstddef>
#include <limits>

#include "scalfmm/meta/traits.hpp"

namespace scalfmm::container
{
    // classic point implementation
    template<typename Arithmetic, std::size_t Dim>
    struct point_impl : public std::array<Arithmetic, Dim>
    {
      public:
        using base_type = std::array<Arithmetic, Dim>;
        /// Floating number type
        using value_type = Arithmetic;
        /// Dimension type
        using dimension_type = std::size_t;
        /// Space dimension count
        constexpr static const std::size_t dimension = Dim;

        point_impl(std::initializer_list<value_type> l) { std::copy(l.begin(), l.end(), this->begin()); }

        point_impl() = default;
        point_impl(point_impl const&) = default;
        point_impl(point_impl&&) noexcept = default;
        auto operator=(point_impl const&) -> point_impl& = default;
        auto operator=(point_impl&&) noexcept -> point_impl& = default;
        ~point_impl() = default;

        point_impl(std::array<value_type, dimension> a)
          : base_type(a)
        {
        }

        explicit point_impl(value_type to_splat)
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = to_splat;
            }
        }

        template<typename U>
        point_impl(point_impl<U, dimension> const& other)
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = value_type(other[i]);
            }
        }

        template<typename U>
        explicit point_impl(U const* other)
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = value_type(other[i]);
            }
        }
    };

    // proxy for point
    template<typename Arithmetic, std::size_t Dim>
    struct point_proxy : public std::array<std::reference_wrapper<Arithmetic>, Dim>
    {
      public:
        using base_type = std::array<std::reference_wrapper<Arithmetic>, Dim>;
        /// Floating number type
        using value_type = std::decay_t<Arithmetic>;
        /// Reference wrapper type
        using reference_wrapper_type = std::reference_wrapper<value_type>;
        using const_reference_wrapper_type = std::reference_wrapper<std::add_const_t<value_type>>;
        /// Dimension type
        using dimension_type = std::size_t;
        /// Space dimension count
        constexpr static const std::size_t dimension = Dim;

        point_proxy() = delete;
        point_proxy(point_proxy const&) = default;
        point_proxy(point_proxy&&) noexcept = default;
        [[nodiscard]] auto operator=(point_proxy const&) -> point_proxy& = default;
        [[nodiscard]] auto operator=(point_proxy&&) noexcept -> point_proxy& = default;
        ~point_proxy() = default;

        explicit point_proxy(std::array<value_type, dimension>& a)
          : base_type(get_reference_sequence(a))
        {
        }

        explicit point_proxy(std::array<value_type, dimension> const& a)
          : base_type(get_reference_sequence(a))
        {
        }

        explicit point_proxy(std::array<reference_wrapper_type, dimension> const& a)
          : base_type(a)
        {
        }
        explicit point_proxy(std::array<const_reference_wrapper_type, dimension> const& a)
          : base_type(a)
        {
        }

        template<typename... Ts, std::enable_if_t<meta::all(std::is_same_v<value_type, Ts>...), int> = 0>
        explicit point_proxy(std::tuple<Ts...>& a)
          : base_type(get_reference_sequence(a))
        {
        }

        template<typename... Ts, std::enable_if_t<meta::all(std::is_same_v<value_type, Ts>...), int> = 0>
        explicit point_proxy(std::tuple<Ts...> const& a)
          : base_type(get_reference_sequence(a))
        {
        }

        // constructor from tuples of references
        template<typename... Ts,
                 std::enable_if_t<
                   meta::all(std::is_same_v<std::add_lvalue_reference_t<value_type>, Ts>...) ||
                     meta::all(std::is_same_v<std::add_lvalue_reference_t<std::add_const_t<value_type>>, Ts>...),
                   int> = 0>
        explicit point_proxy(std::tuple<Ts...>&& a)
          : base_type(get_reference_sequence(a))
        {
        }

        // element access function returning the underlying reference
        [[nodiscard]] constexpr inline auto at(std::size_t pos) -> value_type& { return base_type::at(pos).get(); }
        [[nodiscard]] constexpr inline auto at(std::size_t pos) const -> value_type const& { return base_type::at(pos).get(); }

        [[nodiscard]] constexpr inline auto operator[](std::size_t pos) -> value_type& { return at(pos); }
        [[nodiscard]] constexpr inline auto operator[](std::size_t pos) const -> value_type const& { return at(pos); }
    };

    // entry point and test if the type is arithmetic
    template<typename Arithmetic, std::size_t Dim = 3, typename Enable = void>
    struct point
    {
        static_assert(meta::is_arithmetic<std::decay_t<Arithmetic>>::value, "Point's inner type should be arithmetic!");
    };

    // selection on the template parameter
    // if Arithmetic is a ref with get a proxy, if not a classic point.
    template<typename Arithmetic, std::size_t Dim>
    struct point<Arithmetic, Dim, typename std::enable_if<meta::is_arithmetic<std::decay_t<Arithmetic>>::value>::type>
      : std::conditional_t<std::is_reference_v<Arithmetic>, point_proxy<std::remove_reference_t<Arithmetic>, Dim>,
                           point_impl<Arithmetic, Dim>>
    {
        static constexpr std::size_t dimension{Dim};
        using arithmetic_type = Arithmetic;
        using value_type = std::decay_t<Arithmetic>;
        using point_proxy_type = point_proxy<std::remove_reference_t<arithmetic_type>, dimension>;
        using point_impl_type = point_impl<std::remove_reference_t<arithmetic_type>, dimension>;
        using base_type =
          std::conditional_t<std::is_reference_v<Arithmetic>, point_proxy_type,
                             point_impl_type>;

        using base_type::base_type;

        auto operator=(point<value_type, dimension> p) noexcept
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] = p[i];
            }
            return *this;
        }

        // Addition assignment operator
        template<typename PointOrProxyOrArray>
        inline auto operator+=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] += other[i];
            }
            return *this;
        }

        // Soustraction assignment operator
        template<typename PointOrProxyOrArray>
        inline auto operator-=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] -= other[i];
            }
            return *this;
        }

        // Data to data multiplication assignment
        template<typename PointOrProxyOrArray>
        inline auto operator*=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] *= other[i];
            }
            return *this;
        }

        // Data to data division assignment
        template<typename PointOrProxyOrArray>
        inline auto operator/=(PointOrProxyOrArray const& other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] /= other[i];
            }
            return *this;
        }

        // Addition assignment operator
        inline auto operator+=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] += other;
            }
            return *this;
        }

        // Soustraction assignment operator
        inline auto operator-=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] -= other;
            }
            return *this;
        }

        // Data to data multiplication assignment
        inline auto operator*=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] *= other;
            }
            return *this;
        }

        // Data to data division assignment
        inline auto operator/=(value_type other) -> point&
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                (*this)[i] /= other;
            }
            return *this;
        }

        template<typename A, std::size_t D>
        inline friend auto operator<<(std::ostream& os, const point<A, D>& pos) -> std::ostream&;

        template<typename A, std::size_t D>
        inline friend auto operator+(point other, point const& another)
          -> point<std::decay_t<A>, D>;
        template<typename A, std::size_t D>
        inline friend auto operator-(point other, point const& another)
          -> point<std::decay_t<A>, D>;
        template<typename A, std::size_t D>
        inline friend auto operator*(point other, point const& another)
          -> point<std::decay_t<A>, D>;
        template<typename A, std::size_t D>
        inline friend auto operator/(point other, point const& another)
          -> point<std::decay_t<A>, D>;

        template<typename A, std::size_t D>
        inline friend auto operator+(point other, value_type another)
          -> point<std::decay_t<A>, D>;
        template<typename A, std::size_t D>
        inline friend auto operator-(point other, value_type another)
          -> point<std::decay_t<A>, D>;
        template<typename A, std::size_t D>
        inline friend auto operator*(point other, value_type another)
          -> point<std::decay_t<A>, D>;
        template<typename A, std::size_t D>
        inline friend auto operator/(point other, value_type another)
          -> point<std::decay_t<A>, D>;

        template<typename A, std::size_t D>
        inline friend auto operator+(value_type another, point other)
          -> point<std::decay_t<A>, D>;
        template<typename A, std::size_t D>
        inline friend auto operator*(value_type another, point other)
          -> point<std::decay_t<A>, D>;

        ///
        /// \brief  compute the minimum of the coordinates of point_impl.
        /// \return the minimum of the coordinates.
        ///
        inline auto min() const -> value_type
        {
            value_type min{std::numeric_limits<value_type>::max()};
            for(auto a: *this)
            {
                min =std::min(min, a);
            }
            return min;
        }
                ///
        /// \brief  compute the minimum of the coordinates of point_impl.
        /// \return the minimum of the coordinates.
        ///
        inline auto max() const -> value_type
        {
            value_type max{-std::numeric_limits<value_type>::max()};
            for(auto a: *this)
            {
                max =std::max(max, a);
            }
            return max;
        }

        ///
        /// \brief norm compute the L2 norm of the point_impl.
        /// \return the L2 norm of the point_impl.
        ///
        inline auto norm() const -> value_type
        {
            value_type square_sum{0};
            for(auto a: *this)
            {
                square_sum += a * a;
            }
            return std::sqrt(square_sum);
        }

     ///
        /// \brief norm2 compute the L2 norm squared of the point_impl.
        /// \return the L2 norm squared of the point_impl.
        ///
        inline auto norm2() const -> value_type
        {
            value_type square_sum{0};
            for(auto a: *this)
            {
                square_sum += a * a;
            }
            return square_sum;
        }
        template<typename PointOrProxyOrArray>
        inline auto distance(PointOrProxyOrArray const& p) const -> value_type
        {
            value_type square_sum{0};
            for(std::size_t i{0}; i<dimension; ++i)
            {
                auto tmp = (*this)[i] - p.at(i);
                square_sum += tmp*tmp;
            }
            return std::sqrt(square_sum);
        }
    };

    template<typename Arithmetic, std::size_t Dim>
    inline auto operator<<(std::ostream& os, const point<Arithmetic, Dim>& pos) -> std::ostream&
    {
        os << "[";
        for(std::size_t i{0}; i < Dim - 1; ++i)
        {
            os << pos.at(i) << ", ";
        }
        os << pos.at(Dim - 1) << "]";
        return os;
    }

    // Addition operator
    template<typename A, std::size_t D>
    inline auto operator+(point<A, D> other, point<A, D> const& another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) + another.at(i);
        }
        return res;
    }

    // Soustraction operator
    template<typename A, std::size_t D>
    inline auto operator-(point<A, D> other, point<A, D> const& another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) - another.at(i);
        }
        return res;
    }

    // Multiply operator
    template<typename A, std::size_t D>
    inline auto operator*(point<A, D> other, point<A, D> const& another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) * another.at(i);
        }
        return res;
    }

    // Divide operator
    template<typename A, std::size_t D>
    inline auto operator/(point<A, D> other, point<A, D> const& another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) / another.at(i);
        }
        return res;
    }

    // Addition operator
    template<typename A, std::size_t D>
    inline auto operator+(point<A, D> other, std::decay_t<A> another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) + another;
        }
        return res;
    }

    // Soustraction operator
    template<typename A, std::size_t D>
    inline auto operator-(point<A, D> other, std::decay_t<A> another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) - another;
        }
        return res;
    }

    // Multiply operator
    template<typename A, std::size_t D>
    inline auto operator*(point<A, D> other, std::decay_t<A> another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) * another;
        }
        return res;
    }

    // Divide operator
    template<typename A, std::size_t D>
    inline auto operator/(point<A, D> other, std::decay_t<A> another)
      -> point<std::decay_t<A>, D>
    {
        point<std::decay_t<A>, D> res{};
        for(std::size_t i = 0; i < D; ++i)
        {
            res.at(i) = other.at(i) / another;
        }
        return res;
    }

    // Addition operator
    template<typename A, std::size_t D>
    inline auto operator+(std::decay_t<A> another, point<A, D> other)
      -> point<std::decay_t<A>, D>
    {
        return other + another;
    }

    // Multiply operator
    template<typename A, std::size_t D>
    inline auto operator*(std::decay_t<A> another, point<A, D> other)
      -> point<std::decay_t<A>, D>
    {
        return other * another;
    }

    // Divide operator
    template<typename A, std::size_t D>
    inline auto operator/(std::decay_t<A> another, point<A, D> other)
      -> point<std::decay_t<A>, D>
    {
        point<A,D> val(another);
        return val / other;
    }


//
    //___________________ TODO : ________________________
    ///** Equality test operator */
    // template<class T>
    // inline friend auto operator==(const point_impl<T, Dim>& lhs, const point_impl<T, Dim>& rhs)
    //  -> std::enable_if_t<meta::is_float<T>::value, bool>
    //{
    //    auto lhs_it = lhs.begin();
    //    auto rhs_it = rhs.begin();
    //    const T p{1e-7};
    //    for(std::size_t i = 0; i < Dim; ++i, ++lhs_it, ++rhs_it)
    //    {
    //        if(!meta::feq(*lhs_it, *rhs_it, p))
    //        {
    //            return false;
    //        }
    //    }
    //    return true;
    //}

    ///** Equality test operator */
    // template<class T>
    // inline friend auto operator==(const point_impl<T, Dim>& lhs, const point_impl<T, Dim>& rhs)
    //  -> std::enable_if_t<meta::is_double<T>::value, bool>
    //{
    //    auto lhs_it = lhs.begin();
    //    auto rhs_it = rhs.begin();
    //    const T p{1e-13};
    //    for(std::size_t i = 0; i < Dim; ++i, ++lhs_it, ++rhs_it)
    //    {
    //        if(!meta::feq(*lhs_it, *rhs_it, p))
    //        {
    //            return false;
    //        }
    //    }
    //    return true;
    //}

    ///** Equality test operator */
    // template<class T>
    // inline friend auto operator==(const point_impl<T, Dim>& lhs, const point_impl<T, Dim>& rhs)
    //  -> std::enable_if_t<std::is_integral<T>::value, bool>
    //{
    //    auto lhs_it = lhs.begin();
    //    auto rhs_it = rhs.begin();

    //    for(std::size_t i = 0; i < Dim; i++, ++lhs_it, ++rhs_it)
    //    {
    //        if(*lhs_it != *rhs_it)
    //        {
    //            return false;
    //        }
    //    }
    //    return true;
    //}

    ///** Equality test operator */
    // template<class T>
    // inline friend auto operator==(const point_impl<T, Dim>& lhs, const point_impl<T, Dim>& rhs)
    //  -> std::enable_if_t<meta::is_simd<T>::value, bool>
    //{
    //    auto lhs_it = lhs.begin();
    //    auto rhs_it = rhs.begin();

    //    for(std::size_t i = 0; i < Dim; i++, ++lhs_it, ++rhs_it)
    //    {
    //        auto different = *lhs_it != *rhs_it;

    //        if(!xsimd::any(different))
    //        {
    //            return false;
    //        }
    //    }
    //    return true;
    //}

    /** Non equality test operator */
    //inline friend auto operator!=(const point_impl& lhs, const point_impl& rhs) { return !(lhs == rhs); }
}   // end of namespace scalfmm::container

#endif
