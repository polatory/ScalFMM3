#ifndef SCALFMM_UTILS_OSTREAM_TUPLE_HPP
#define SCALFMM_UTILS_OSTREAM_TUPLE_HPP

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>

#include "inria/integer_sequence.hpp"
#include "inria/ostream_joiner.hpp"

#include "scalfmm/meta/utils.hpp"
namespace scalfmm
{
    namespace details
    {
        namespace tuple_helper
        {
            /** \brief Helper for tuple formatted output
             *
             * \param os Output stream
             * \param t  Printed tuple
             * \param index_sequence unnamed parameter for automatic deduction of
             * Indices template parameter
             *
             * \tparam Types Types contained in the tuple, automatically deduced
             * \tparam Indices Type indices, automatically deduced
             */
            template<typename... Types, std::size_t... Indices>
            inline void formated_output_old(std::ostream& os, const std::tuple<Types...>& t,
                                        inria::index_sequence<Indices...>)
            {
                os << "(";   // opening brace
                             // initializer_list members are evaluated from left to right; this
                             // evaluates to an equivalent to:
                             //
                             //     os << std::get<0>(t) << ", ";
                             //     os << std::get<1>(t) << ", "
                             //     ...
                             //     std::get<N>(t) << "";
                //   auto l = {(os << std::get<Indices>(t) << (Indices != sizeof...(Types) - 1 ? ", " : ""), 0)...};
                //  (void)l;     // ignore fact that initializer_list is not used
                (..., (os << (Indices == 0 ? "" : ", ") << std::get<Indices>(t)));
                os << ")";   // closing brace
            }
            /**
             * @brief Print formated tuple
             * 
             * @tparam Tuple 
             * @tparam Indices 
             * @param os the stream 
             * @param t the tuple
             */
            template<typename  Tuple, std::size_t... Indices>
            inline void formated_output(std::ostream& os, const Tuple& t,
                                        inria::index_sequence<Indices...>)
            {
                os << "[";  
                (..., (os << (Indices == 0 ? "" : ", ") << std::get<Indices>(t)));
                os << "]";   // closing brace
            }
            template<class... Ts>
            struct tuple_wrapper
            {
                const std::tuple<Ts...>& t;
                friend std::ostream& operator<<(std::ostream& os, const tuple_wrapper& w)
                {
                    using details::tuple_helper::formated_output;
                    formated_output(os, w.t, inria::index_sequence_for<Ts...>{});
                    return os;
                }
            };

        }   // namespace tuple_helper
    }       // namespace details

    namespace
    {
        constexpr struct
        {
            template<class... Ts>
            details::tuple_helper::tuple_wrapper<Ts...> operator()(const std::tuple<Ts...>& t) const
            {
                return {t};
            }
        } tuple_out{};

    }   // namespace

    namespace details
    {
        /**
         * \brief Silence the `unused variable` warnings about tuple_out
         */
        template<class T>
        void silence_tuple_out_warning()
        {
            tuple_out(std::tuple<>{});
        }

    }   // namespace details

    namespace io
    {
        ///
        /// \brief operator << for array std::array<T,N>
        /// 
        /// using namespace io ; 
        /// std::array(int, 2) a; 
        /// std::cout << a << std::endl ; 
        /// print array [a_1, ..., a_N]
        /// \param os ostream
        /// \param array to print
        ///
        template<typename T, std::size_t N>
        inline auto operator<<(std::ostream& os, const std::array<T, N>& array) -> std::ostream&
        {
            os << "[";
            for(auto it = array.begin(); it != array.end() - 1; it++)
            {
                os << *it << ", ";
            }
            os << array.back() << "]";
            return os;
        }
        ///
        /// \brief print a vector and its size
        ///
        /// The output is
        ///  title (size) values
        ///
        /// The values are separated by a white space
        ///
        /// \param[in] title
        /// \param[in] v a vector (need size method)
        ///
        template<typename Vector_type>
        void print(const std::string&& title, Vector_type const& v)
        {
            std::cout << title << " (" << v.size() << ") ";
            if(v.size() > 0)
            {
                for(auto& i: v)
                    std::cout << i << ' ';
            }
            std::cout << '\n';
        };

        ///
        /// \brief print print a vector and its size starting at first and ending at end
        ///
        /// The output is
        ///  title (size) [value1 value2 ...]
        ///
        /// The values are separated by a white space
        ///
        /// \param title string to print
        /// \param first begin iterator
        /// \param last  end iterator
        ///
        template<typename Iterator_type>
        void print(std::ostream& out, const std::string&& title, Iterator_type first, Iterator_type last,
                   std::string&& sep = " ")
        {
            auto size = std::distance(first, last);

            out << title << " (" << size << ") [";
            if(size)
            {
                auto lastm1 = --last;
                Iterator_type it;
                for(it = first; it != lastm1; ++it)
                    out << *it << sep;
                out << *it;
            }
            out << ']';
        };
        template<typename Vector_type>
        void print(std::ostream& out, const std::string&& title, Vector_type& v, std::string&& sep = " ")
        {
            print(std::cout, std::move(title), v.begin(), v.end(), std::move(sep));
            std::cout << std::endl;
        };
        /**
         * @brief Print a part or a full array
         *
         * print the array array[0:size]
         *
         * @tparam ARRAY type of teh array
         * @param array the array to print
         * @param size  the list of elements to print
         */
        template<typename ARRAY>
        auto print_array(ARRAY const& array, const int& size) -> void
        {
            std::cout << "[ "<< array[0];
            for(std::size_t i{1}; i < size; ++i)
            {
                std::cout << ", "<< array[i] ;
            }
            std::cout << " ]";
        }
         template<typename T, std::size_t N>
        void print(std::ostream& out,  std::array<T,N>const & a)
         {
            std::cout << "[";
            for(std::size_t i{0}; i < N; ++i)
            {
                std::cout << (i==0 ? "" : ", ") << a[i] ;
            }
            std::cout << "]";
        }
        /**
         * @brief print a tuple 
         * 
         * std::tuple<int, double, int> a{1,0.3,8};
         *  io::print(a) 
         *  the output is [1, 0.3, 8]
         * 
         * @param out the stream
         * @param tup the tuple
         */
         
         template<class... T>
        void print(std::ostream& out, const std::tuple<T...>& tup)
        {
            scalfmm::details::tuple_helper::formated_output(out, tup, std::make_index_sequence<sizeof...(T)>());
        }
        /**
         * @brief print a sequence (tuple, array)
         *  tuple<int, double> t{3,0.5}
         *  std::cout << t << std::endl shows [3, 0.5, ]
         *
         * @param[inout] out the stream
         * @param[in] seq the sequence to print
         * @return auto
         */
        template<typename Seq>
        inline auto print_seq(std::ostream& out, Seq const& seq)
        {
            out << "[";
            int i{0};
            meta::for_each(seq, [&out, &i](const auto& s) { out << (i++==0 ? "":", ")<< s ; });  
                      out << "]";
        }
        /**
         * @brief print the address ot the element of a sequence (tuple, array)
         *
         *  tuple<int, double> t{3,0.5}
         *  std::cout << t << std::endl
         *
         * @param[inout] out the stream
         * @param[in] seq the sequence to print
         * @return auto
         */
        template<typename Seq>
        inline auto print_ptr_seq(std::ostream& out, const Seq& seq)
        {
            out << "[";
            int i{0};
            meta::for_each(seq, [&out, &i](const auto& s) { out << (i++==0 ? "":", ")<< &s ; });
            out << "]";
        }

        // template<class... T>
        // inline auto operator<<(std::ostream& os,  std::tuple<T...>const & t) -> std::ostream&
        // {
        //     meta::td<decltype(t)> tt;
        //     scalfmm::details::tuple_helper::formated_output_n(os, t, std::make_index_sequence<sizeof...(T)>());
        //     return os;
        // }

        // template<typename... Args>
        // inline auto print(std::ostream& out, Args&&... args)
        // {
        //     ((out << ' ' << std::forward<Args>(args)), ...);
        // }

    }   // namespace io

}   // namespace scalfmm

#endif
