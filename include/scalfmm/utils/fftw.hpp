// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/fftw.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_FFTW_HPP
#define SCALFMM_UTILS_FFTW_HPP

#include "scalfmm/utils/massert.hpp"

//#include "xtensor-fftw/basic.hpp"
//#include "xtensor-fftw/common.hpp"
//#include "xtensor-fftw/xtensor-fftw_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xcontainer.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xstorage.hpp"
#include "xtensor/xtensor_forward.hpp"
#include "xtl/xcomplex.hpp"

#include <fftw3.h>

#include <algorithm>
#include <any>
#include <complex>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iterator>
#include <mutex>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <type_traits>
#include <valarray>
#include <vector>

// Define if the library is going to be using exceptions.
#if (!defined(__cpp_exceptions) && !defined(__EXCEPTIONS) && !defined(_CPPUNWIND))
#undef XTENSOR_FFTW_DISABLE_EXCEPTIONS
#define XTENSOR_FFTW_DISABLE_EXCEPTIONS
#endif

// Exception support.
#if defined(XTENSOR_FFTW_DISABLE_EXCEPTIONS)
#include <iostream>
#define XTENSOR_FFTW_THROW(_, msg)       \
    {                                    \
      std::cerr << msg << std::endl;     \
      std::abort();                      \
    }
#else
#define XTENSOR_FFTW_THROW(exception, msg) throw exception(msg)
#endif

namespace  xt::fftw
{
          // output to DFT-dimensions conversion
        template<typename output_t>
        inline auto dft_dimensions_from_output(const xt::xarray<output_t, xt::layout_type::row_major>& output,
                                               bool half_plus_one_out, bool odd_last_dim = false)
        {
            auto dft_dimensions = output.shape();

            if(half_plus_one_out)
            {   // r2c
                auto n = dft_dimensions.size();
                if(!odd_last_dim)
                {
                    dft_dimensions[n - 1] = (dft_dimensions[n - 1] - 1) * 2;
                }
                else
                {
                    dft_dimensions[n - 1] = (dft_dimensions[n - 1] - 1) * 2 + 1;
                }
            }

            return dft_dimensions;
        }
}
namespace scalfmm::fftw
{
    /**
     * @brief
     *
     * @tparam ValueType
     * @tparam dim
     */
    template<typename ValueType, std::size_t dim>
    struct fft;

    /**
     * @brief
     *
     * @tparam dim
     */
    template<std::size_t dim>
    struct fft<double, dim>
    {
        using value_type = double;
        using fft_type = xt::xarray<double>;
        using transformed_fft_type = xt::xarray<std::complex<double>>;

        /**
         * @brief
         *
         * @return std::mutex&
         */
        inline std::mutex& fftw_global_mutex()
        {
            static std::mutex m;
            return m;
        }

        /**
         * @brief Construct a new fft object
         *
         */
        fft() = default;

        /**
         * @brief Construct a new fft object
         *
         */
        fft(fft const&) = default;

        /**
         * @brief Construct a new fft object
         *
         */
        fft(fft&&) = delete;

        /**
         * @brief
         *
         * @return fft&
         */
        inline auto operator=(fft const&) -> fft& = default;

        /**
         * @brief
         *
         * @return fft&
         */
        inline auto operator=(fft&&) noexcept -> fft& = delete;

        /**
         * @brief
         *
         * @param order
         */
        auto initialize(std::size_t order) -> void
        {
            std::vector<std::size_t> input_forward_shape(dim, (2 * order - 1));
            std::vector<std::size_t> output_forward_shape(dim, (2 * order - 1));
            output_forward_shape.at(dim - 1) = order;
            m_real_buffer.resize(input_forward_shape);
            m_complex_buffer.resize(output_forward_shape);
            create_plan();
            create_inverse_plan(true);
        }

        /**
         * @brief Create a plan object
         *
         */
        inline auto create_plan() -> void
        {
            if(!plan_exists)
            {
                using fftw_input_t = double;
                using fftw_output_t = fftw_complex;

                bool odd_last_dim = (m_real_buffer.shape()[m_real_buffer.shape().size() - 1] % 2 != 0);

                auto dft_dimensions_unsigned =
                  xt::fftw::dft_dimensions_from_output(m_complex_buffer, true, odd_last_dim);
                std::vector<int> dft_dimensions;
                dft_dimensions.reserve(dft_dimensions_unsigned.size());
                std::transform(dft_dimensions_unsigned.begin(), dft_dimensions_unsigned.end(),
                               std::back_inserter(dft_dimensions), [&](std::size_t d) { return static_cast<int>(d); });

                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                plan = fftw_plan_dft_r2c(
                  static_cast<int>(dim), dft_dimensions.data(),
                  const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_real_buffer.data())),
                  reinterpret_cast<fftw_output_t*>(m_complex_buffer.data()), FFTW_ESTIMATE);
                if(plan == nullptr)
                {
                    XTENSOR_FFTW_THROW(
                      std::runtime_error,
                      "Plan creation returned nullptr. This usually means FFTW cannot create a plan for "
                      "the given arguments "
                      "(e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
                }
                plan_exists = true;
            }
        }

        /**
         * @brief Create a inverse plan object
         *
         * @param odd_last_dim
         */
        inline auto create_inverse_plan(bool odd_last_dim = false) -> void
        {
            if(!plan_inv_exists)
            {
                using fftw_input_t = fftw_complex;
                using fftw_output_t = double;

                auto dft_dimensions_unsigned = xt::fftw::dft_dimensions_from_output(m_real_buffer, false, odd_last_dim);
                std::vector<int> dft_dimensions;
                dft_dimensions.reserve(dft_dimensions_unsigned.size());
                std::transform(dft_dimensions_unsigned.begin(), dft_dimensions_unsigned.end(),
                               std::back_inserter(dft_dimensions), [&](std::size_t d) { return static_cast<int>(d); });

                this->N_dft = static_cast<fftw_output_t>(std::accumulate(
                  dft_dimensions.begin(), dft_dimensions.end(), static_cast<size_t>(1u), std::multiplies<size_t>()));

                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                plan_inv = fftw_plan_dft_c2r(
                  static_cast<int>(dim), dft_dimensions.data(),
                  const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_complex_buffer.data())),
                  reinterpret_cast<fftw_output_t*>(m_real_buffer.data()), FFTW_ESTIMATE);

                if(plan_inv == nullptr)
                {
                    XTENSOR_FFTW_THROW(
                      std::runtime_error,
                      "Plan creation returned nullptr. This usually means FFTW cannot create a plan for "
                      "the given arguments "
                      "(e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
                }
                plan_inv_exists = true;
            }
        }

        /**
         * @brief
         *
         * @param input
         * @param output
         */
        inline auto execute_plan(xt::xarray<double> const& input, xt::xarray<std::complex<double>>& output) -> void
        {
            using fftw_input_t = double;
            using fftw_output_t = fftw_complex;

            if(plan == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_real_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_real_buffer.begin());
            fftw_execute_dft_r2c(plan,
                                 const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_real_buffer.data())),
                                 reinterpret_cast<fftw_output_t*>(m_complex_buffer.data()));
            assertm(m_complex_buffer.size() == output.size(),
                    "Output complex buffer does not have the same size as the fft handler buffer!");
            std::copy(m_complex_buffer.begin(), m_complex_buffer.end(), output.begin());
        }

        /**
         * @brief
         *
         * @param input
         */
        inline auto execute_plan(xt::xarray<double> const& input) -> void
        {
            using fftw_input_t = double;
            using fftw_output_t = fftw_complex;

            if(plan == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_real_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_real_buffer.begin());
            fftw_execute_dft_r2c(plan,
                                 const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_real_buffer.data())),
                                 reinterpret_cast<fftw_output_t*>(m_complex_buffer.data()));
        }

        /**
         * @brief
         *
         * @param input
         * @param output
         */
        inline auto execute_inverse_plan(xt::xarray<std::complex<double>> const& input,
                                         xt::xarray<double>& output) -> void
        {
            using fftw_input_t = fftw_complex;
            using fftw_output_t = double;

            if(plan_inv == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_complex_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_complex_buffer.begin());
            fftw_execute_dft_c2r(
              plan_inv, const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_complex_buffer.data())),
              reinterpret_cast<fftw_output_t*>(m_real_buffer.data()));
            m_real_buffer /= N_dft;
            assertm(m_real_buffer.size() == output.size(),
                    "Output complex buffer does not have the same size as the fft handler buffer!");
            std::copy(m_real_buffer.begin(), m_real_buffer.end(), output.begin());
        }

        /**
         * @brief
         *
         * @param input
         */
        inline auto execute_inverse_plan(xt::xarray<std::complex<double>> const& input) -> void
        {
            using fftw_input_t = fftw_complex;
            using fftw_output_t = double;

            if(plan_inv == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_complex_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_complex_buffer.begin());
            fftw_execute_dft_c2r(
              plan_inv, const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_complex_buffer.data())),
              reinterpret_cast<fftw_output_t*>(m_real_buffer.data()));
            m_real_buffer /= N_dft;
        }

        /**
         * @brief
         *
         */
        inline auto destroy_plan() -> void
        {
            if(plan_exists)
            {
                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                fftw_destroy_plan(plan);
                plan_exists = false;
            }
        }

        /**
         * @brief
         *
         */
        inline auto destroy_inverse_plan() -> void
        {
            if(plan_inv_exists)
            {
                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                fftw_destroy_plan(plan_inv);
                plan_inv_exists = false;
            }
        }

        /**
         * @brief
         *
         * @return fft_type const&
         */
        [[nodiscard]] inline auto real_buffer() const -> fft_type const& { return m_real_buffer; }

        /**
         * @brief
         *
         * @return fft_type const&
         */
        [[nodiscard]] inline auto creal_buffer() const -> fft_type const& { return m_real_buffer; }

        /**
         * @brief
         *
         * @return fft_type&
         */
        [[nodiscard]] inline auto real_buffer() -> fft_type& { return m_real_buffer; }

        /**
         * @brief
         *
         * @return transformed_fft_type const&
         */
        [[nodiscard]] inline auto complex_buffer() const -> transformed_fft_type const& { return m_complex_buffer; }

        /**
         * @brief
         *
         * @return transformed_fft_type const&
         */
        [[nodiscard]] inline auto ccomplex_buffer() const -> transformed_fft_type const& { return m_complex_buffer; }

        /**
         * @brief
         *
         * @return transformed_fft_type&
         */
        [[nodiscard]] inline auto complex_buffer() -> transformed_fft_type& { return m_complex_buffer; }

        /**
         * @brief Destroy the fft object
         *
         */
        ~fft()
        {
            destroy_plan();
            destroy_inverse_plan();
        }

        /**
         * @brief
         *
         */
        fft_type m_real_buffer{};

        /**
         * @brief
         *
         */
        transformed_fft_type m_complex_buffer{};

        /**
         * @brief
         *
         */
        fftw_plan plan;

        /**
         * @brief
         *
         */
        fftw_plan plan_inv;

        /**
         * @brief
         *
         */
        double N_dft{1};

        /**
         * @brief
         *
         */
        bool plan_exists{false};

        /**
         * @brief
         *
         */
        bool plan_inv_exists{false};
    };

    /**
     * @brief
     *
     * @tparam dim
     */
    template<std::size_t dim>
    struct fft<float, dim>
    {
        using value_type = float;
        using fft_type = xt::xarray<float>;
        using transformed_fft_type = xt::xarray<std::complex<float>>;

        /**
         * @brief
         *
         * @return std::mutex&
         */
        inline std::mutex& fftw_global_mutex()
        {
            static std::mutex m;
            return m;
        }

        /**
         * @brief Construct a new fft object
         *
         */
        fft() = default;

        /**
         * @brief Construct a new fft object
         *
         */
        fft(fft const&) = delete;

        /**
         * @brief Construct a new fft object
         *
         */
        fft(fft&&) = delete;

        /**
         * @brief
         *
         * @return fft&
         */
        inline auto operator=(fft const&) -> fft& = delete;

        /**
         * @brief
         *
         * @return fft&
         */
        inline auto operator=(fft&&) noexcept -> fft& = delete;

        /**
         * @brief
         *
         * @param order
         */
        auto initialize(std::size_t order) -> void
        {
            std::vector<std::size_t> input_forward_shape(dim, (2 * order - 1));
            std::vector<std::size_t> output_forward_shape(dim, (2 * order - 1));
            output_forward_shape.at(dim - 1) = order;
            m_real_buffer.resize(input_forward_shape);
            m_complex_buffer.resize(output_forward_shape);
            create_plan();
            create_inverse_plan(true);
        }

        /**
         * @brief Create a plan object
         *
         */
        inline auto create_plan() -> void
        {
            if(!plan_exists)
            {
                using fftw_input_t = float;
                using fftw_output_t = fftwf_complex;

                bool odd_last_dim = (m_real_buffer.shape()[m_real_buffer.shape().size() - 1] % 2 != 0);

                auto dft_dimensions_unsigned =
                  xt::fftw::dft_dimensions_from_output(m_complex_buffer, true, odd_last_dim);
                std::vector<int> dft_dimensions;
                dft_dimensions.reserve(dft_dimensions_unsigned.size());
                std::transform(dft_dimensions_unsigned.begin(), dft_dimensions_unsigned.end(),
                               std::back_inserter(dft_dimensions), [&](std::size_t d) { return static_cast<int>(d); });

                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                plan = fftwf_plan_dft_r2c(
                  static_cast<int>(dim), dft_dimensions.data(),
                  const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_real_buffer.data())),
                  reinterpret_cast<fftw_output_t*>(m_complex_buffer.data()), FFTW_ESTIMATE);
                if(plan == nullptr)
                {
                    XTENSOR_FFTW_THROW(
                      std::runtime_error,
                      "Plan creation returned nullptr. This usually means FFTW cannot create a plan for "
                      "the given arguments "
                      "(e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
                }
                plan_exists = true;
            }
        }

        /**
         * @brief Create a inverse plan object
         *
         * @param odd_last_dim
         */
        inline auto create_inverse_plan(bool odd_last_dim = false) -> void
        {
            if(!plan_inv_exists)
            {
                using fftw_input_t = fftwf_complex;
                using fftw_output_t = float;

                auto dft_dimensions_unsigned = xt::fftw::dft_dimensions_from_output(m_real_buffer, false, odd_last_dim);
                std::vector<int> dft_dimensions;
                dft_dimensions.reserve(dft_dimensions_unsigned.size());
                std::transform(dft_dimensions_unsigned.begin(), dft_dimensions_unsigned.end(),
                               std::back_inserter(dft_dimensions), [&](std::size_t d) { return static_cast<int>(d); });

                this->N_dft = static_cast<fftw_output_t>(std::accumulate(
                  dft_dimensions.begin(), dft_dimensions.end(), static_cast<size_t>(1u), std::multiplies<size_t>()));

                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                plan_inv = fftwf_plan_dft_c2r(
                  static_cast<int>(dim), dft_dimensions.data(),
                  const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_complex_buffer.data())),
                  reinterpret_cast<fftw_output_t*>(m_real_buffer.data()), FFTW_ESTIMATE);

                if(plan_inv == nullptr)
                {
                    XTENSOR_FFTW_THROW(
                      std::runtime_error,
                      "Plan creation returned nullptr. This usually means FFTW cannot create a plan for "
                      "the given arguments "
                      "(e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
                }
                plan_inv_exists = true;
            }
        }

        /**
         * @brief
         *
         * @param input
         * @param output
         */
        inline auto execute_plan(xt::xarray<float> const& input, xt::xarray<std::complex<float>>& output) -> void
        {
            using fftw_input_t = float;
            using fftw_output_t = fftwf_complex;

            if(plan == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_real_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_real_buffer.begin());
            fftwf_execute_dft_r2c(
              plan, const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_real_buffer.data())),
              reinterpret_cast<fftw_output_t*>(m_complex_buffer.data()));
            assertm(m_complex_buffer.size() == output.size(),
                    "Output complex buffer does not have the same size as the fft handler buffer!");
            std::copy(m_complex_buffer.begin(), m_complex_buffer.end(), output.begin());
        }

        /**
         * @brief
         *
         * @param input
         */
        inline auto execute_plan(xt::xarray<float> const& input) -> void
        {
            using fftw_input_t = float;
            using fftw_output_t = fftwf_complex;

            if(plan == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_real_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_real_buffer.begin());
            fftwf_execute_dft_r2c(
              plan, const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_real_buffer.data())),
              reinterpret_cast<fftw_output_t*>(m_complex_buffer.data()));
        }

        /**
         * @brief
         *
         * @param input
         * @param output
         */
        inline auto execute_inverse_plan(xt::xarray<std::complex<float>> const& input,
                                         xt::xarray<float>& output) -> void
        {
            using fftw_input_t = fftwf_complex;
            using fftw_output_t = float;

            if(plan_inv == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_complex_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_complex_buffer.begin());
            fftwf_execute_dft_c2r(
              plan_inv, const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_complex_buffer.data())),
              reinterpret_cast<fftw_output_t*>(m_real_buffer.data()));
            m_real_buffer /= N_dft;
            assertm(m_real_buffer.size() == output.size(),
                    "Output complex buffer does not have the same size as the fft handler buffer!");
            std::copy(m_real_buffer.begin(), m_real_buffer.end(), output.begin());
        }

        /**
         * @brief
         *
         * @param input
         */
        inline auto execute_inverse_plan(xt::xarray<std::complex<float>> const& input) -> void
        {
            using fftw_input_t = fftwf_complex;
            using fftw_output_t = float;

            if(plan_inv == nullptr)
            {
                XTENSOR_FFTW_THROW(
                  std::runtime_error,
                  "Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given "
                  "arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
            }
            assertm(input.size() == m_complex_buffer.size(),
                    "Input real buffer does not have the same size as the fft handler buffer!");
            std::copy(input.begin(), input.end(), m_complex_buffer.begin());
            fftwf_execute_dft_c2r(
              plan_inv, const_cast<fftw_input_t*>(reinterpret_cast<const fftw_input_t*>(m_complex_buffer.data())),
              reinterpret_cast<fftw_output_t*>(m_real_buffer.data()));
            m_real_buffer /= N_dft;
        }

        /**
         * @brief
         *
         */
        inline auto destroy_plan() -> void
        {
            if(plan_exists)
            {
                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                fftwf_destroy_plan(plan);
                plan_exists = false;
            }
        }

        /**
         * @brief
         *
         */
        inline auto destroy_inverse_plan() -> void
        {
            if(plan_inv_exists)
            {
                std::lock_guard<std::mutex> guard(fftw_global_mutex());
                fftwf_destroy_plan(plan_inv);
                plan_inv_exists = false;
            }
        }

        /**
         * @brief
         *
         * @return fft_type const&
         */
        [[nodiscard]] inline auto real_buffer() const -> fft_type const& { return m_real_buffer; }

        /**
         * @brief
         *
         * @return fft_type const&
         */
        [[nodiscard]] inline auto creal_buffer() const -> fft_type const& { return m_real_buffer; }

        /**
         * @brief
         *
         * @return fft_type&
         */
        [[nodiscard]] inline auto real_buffer() -> fft_type& { return m_real_buffer; }

        /**
         * @brief
         *
         * @return transformed_fft_type const&
         */
        [[nodiscard]] inline auto complex_buffer() const -> transformed_fft_type const& { return m_complex_buffer; }

        /**
         * @brief
         *
         * @return transformed_fft_type&
         */
        [[nodiscard]] inline auto complex_buffer() -> transformed_fft_type& { return m_complex_buffer; }

        /**
         * @brief Destroy the fft object
         *
         */
        ~fft()
        {
            destroy_plan();
            destroy_inverse_plan();
        }

        /**
         * @brief
         *
         */
        fft_type m_real_buffer{};

        /**
         * @brief
         *
         */
        transformed_fft_type m_complex_buffer{};

        /**
         * @brief
         *
         */
        fftwf_plan plan;

        /**
         * @brief
         *
         */
        fftwf_plan plan_inv;

        /**
         * @brief
         *
         */
        float N_dft{1};

        /**
         * @brief
         *
         */
        bool plan_exists{false};

        /**
         * @brief
         *
         */
        bool plan_inv_exists{false};
    };
}   // namespace scalfmm::fftw

#endif   // SCALFMM_UTILS_FFTW_HPP
