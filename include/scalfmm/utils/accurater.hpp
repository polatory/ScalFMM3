// See LICENCE file at project root
#ifndef SCALFMM_UTILS_ACCURATER_HPP
#define SCALFMM_UTILS_ACCURATER_HPP

#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdlib>

/**
 * \class A class to compute accuracy  between data
 *
 *
 */
namespace scalfmm
{
    namespace utils
    {
        template<class real_type>
        class accurater
        {
            std::size_t _nb_elements{};   ///< number of elements used to compute the error
            real_type _l2_norm{};         ///< l2-norm of the reference value
            real_type _l2_diff{};         ///< l2-norm of the difference between the reference value and the value
            real_type _max{};             ///< infinity norm of the reference value
            real_type _max_diff{};   ///<  infinity norm of the difference between the reference value and the value
          public:
            accurater() = default;
            /** with inital values */
            accurater(const real_type ref_value[], const real_type value[], const std::size_t& nbValues)
            {
                this->add(ref_value, value, nbValues);
            }

            /**
             * @brief Add |ref_value-value| to the current accurater
             *
             * @param ref_value
             * @param value
             */

            void add(const real_type& ref_value, const real_type& value)
            {
                _l2_diff += (value - ref_value) * (value - ref_value);
                _l2_norm += ref_value * ref_value;
                _max = std::max(_max, std::abs(ref_value));
                _max_diff = std::max(_max_diff, std::abs(ref_value - value));
                ++_nb_elements;
            }
            /**
             * @brief Add all vector elements |ref_value[i]-value[i]| to the current accurater
             *
             * @param ref_value reference array
             * @param value  array to check
             */
            void add(const real_type* const ref_values, const real_type* const values, const std::size_t& nb_values)
            {
                for(std::size_t idx = 0; idx < nb_values; ++idx)
                {
                    this->add(ref_values[idx], values[idx]);
                }
                _nb_elements += nb_values;
            }
            template<class vector_type>
            void add(const vector_type& ref_values, const vector_type& values)
            {
                if(values->size() != ref_values->size())
                {
                    std::cerr << "Wrong size in add method. " << ref_values.size() << " != " << values << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                this->add(ref_values->data(), values->data(), values->size());
            }
            /** Add an accurater*/
            void add(const accurater& inAcc)
            {
                _l2_diff += inAcc.get_l2_diff_norm();
                _l2_norm += inAcc.get_l2_norm();
                _max = std::max(_max, inAcc.get_max());
                _max_diff = std::max(_max_diff, inAcc.get_infinity_norm());
                _nb_elements += inAcc.get_nb_elements();
            }
            /**
             * @brief Get the the square of the l2 norm of the error sum_i(ref_i-val_i)^2)
             *
             * @return real_type
             */
            real_type get_l2_diff_norm2() const { return _l2_diff; }
            /**
             * @brief Get the l2 norm of the error sqrt( sum_i(ref_i-val_i)^2)
             *
             * @return real_type
             */
            real_type get_l2_norm() const { return std::sqrt(_l2_diff); }
            /**
             * @brief Get the l2 norm of the reference vector sqrt( sum_i(ref_i)^2)
             *
             * @return real_type
             */
            real_type get_ref_l2_norm() const { return std::sqrt(_l2_norm); }
            /**
             * @brief Get the maximum value of the reference max_i(|ref_i|)
             *
             * @return real_type
             */
            real_type get_ref_max() const { return _max; }

            /**
             * @brief Get the nb elements used in the norm computation
             *
             * @return auto
             */
            auto get_nb_elements() const { return _nb_elements; }

            void set_nb_elements(const std::size_t& n) { _nb_elements = n; }

            /** Get the root mean squared error*/
            real_type get_mean_squared_error() const { return std::sqrt(_l2_diff); }
            /** Get the root-mean-square error  */
            real_type get_rms_error() const { return std::sqrt(_l2_diff / static_cast<real_type>(_nb_elements)); }
            /**
             * @brief Get the infinity norm of the error max_i|ref_i-val_i|
             *
             * @return real_type
             */
            real_type get_infinity_norm() const { return _max_diff; }
            /**
             * @brief Get the relative L2 norm of the error  sqrt( sum_i(ref_i-val_i)^2/sum_i(ref_i^2))
             *
             * @return real_type
             */
            real_type get_relative_l2_norm() const { return std::sqrt(_l2_diff / _l2_norm); }
            /**
             * @brief Get the relative infinity norm of the error  max_i|ref_i-val_i|/max_i|ref_i|
             *
             * @return real_type
             */
            real_type get_relative_infinity_norm() const { return _max_diff / _max; }
            /**
             * @brief Print the relative errors (L2, RMS, Infinity)
             *
             * @param output  the stream
             * @param inAccurater  the accurater containing the errors to print
             *
             */
            template<class StreamClass>
            friend StreamClass& operator<<(StreamClass& output, const accurater& inAccurater)
            {
                output << "[Error] Relative L2-norm = " << inAccurater.get_relative_l2_norm()
                       << " \t RMS norm = " << inAccurater.get_rms_error()
                       << " \t Relative  infinity norm = " << inAccurater.get_relative_infinity_norm();
                return output;
            }
            /**
             * @brief reset the accurater
             *
             */
            void reset()
            {
                _l2_norm = real_type(0);
                _l2_diff = real_type(0);
                _max = real_type(0);
                _max_diff = real_type(0);
                _nb_elements = 0;
            }
        };

    }   // namespace utils
}   // namespace scalfmm

#endif   // SCALFMM_UTILS_ACCURATER_HPP
