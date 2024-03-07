#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/compare_results.hpp"
#include "scalfmm/utils/tensor.hpp"
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <ios>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

#include <cmath>
#include <scalfmm/container/point.hpp>
#include <scalfmm/interpolation/mapping.hpp>
#include <scalfmm/tools/colorized.hpp>
#include <scalfmm/utils/math.hpp>
#include <scalfmm/utils/timer.hpp>
#include <xsimd/types/xsimd_base.hpp>
#include <xsimd/types/xsimd_traits.hpp>
#include <xsimd/xsimd.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xtensor_forward.hpp>
#include <xtensor/xtensor_simd.hpp>
#include <xtensor/xutils.hpp>
#include <xtl/xcomplex.hpp>

using namespace scalfmm;
template<typename MatrixKernel, typename ValueType, std::size_t Dimension, std::size_t SavedDimension>
struct build
{
    using matrix_kernel_type = MatrixKernel;
    using value_type = ValueType;

    template<typename TensorViewX, typename TensorViewY>
    [[nodiscard]] inline auto operator()(TensorViewX&& X, TensorViewY&& Y, std::size_t order)
    {
        static constexpr std::size_t current_dimension = Dimension;
        static constexpr std::size_t saved_dimension = SavedDimension;
        static constexpr std::size_t dimension_decrease = current_dimension - 1;
        std::size_t ntilde = (2 * order) - 1;
        xt::xarray<std::string> c(std::vector(current_dimension, std::size_t(ntilde)));
        build<matrix_kernel_type, value_type, dimension_decrease, saved_dimension> build_c1{};

        // auto range = xt::all();
        // for(std::size_t i = 0; i<order; ++i)
        //{
        //    auto c1 = tensor::get_view<dimension_decrease>(c, i, range, tensor::row{});
        //    auto X_view =
        //      tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), i, range, tensor::column{});
        //    auto Y_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), 0, range, tensor::row{});
        //    c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order);
        //}
        for(std::size_t i = 0; i < order; ++i)
        {
            auto range = xt::all();
            auto c1 = tensor::get_view<dimension_decrease>(c, i, range, tensor::row{});
            auto X_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), i, range, tensor::row{});
            // auto X_view = tensor::gather<current_dimension>(X, i);
            std::cout << "X_view=" << X_view << '\n';
            // auto Y_view = tensor::gather<current_dimension>(Y, 0);
            auto Y_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), 0, range, tensor::row{});
            // auto Y_view = xt::view(Y, xt::all(), xt::keep(0));
            // std::cout << "Y_view=" << Y_view << '\n';
            // tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), 0, range, tensor::row{});
            c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order);
        }

        for(std::size_t i = 1; i < order; ++i)
        {
            auto range = xt::all();
            auto c1 = tensor::get_view<dimension_decrease>(c, order - 1 + i, range, tensor::row{});
            auto X_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), 0, range, tensor::row{});
            // auto X_view = tensor::gather<current_dimension>(X, 0);
            std::cout << "X_view=" << X_view << '\n';
            // auto Y_view = tensor::gather<current_dimension>(Y, order-i);
            auto Y_view =
              tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), order - i, range, tensor::row{});
            // auto Y_view = xt::view(Y, xt::all(), xt::keep(order-i));
            std::cout << "Y_view=" << Y_view << '\n';
            // tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), order-i, range, tensor::row{});
            // //column
            c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order);
        }
        return c;
    }
};

template<typename MatrixKernel, typename ValueType, std::size_t SavedDimension>
struct build<MatrixKernel, ValueType, 1, SavedDimension>
{
    using matrix_kernel_type = MatrixKernel;
    using value_type = ValueType;

    template<typename TensorViewX, typename TensorViewY>
    [[nodiscard]] inline auto operator()(TensorViewX&& X, TensorViewY&& Y, std::size_t order)
    {
        std::size_t ntilde = 2 * order - 1;
        xt::xarray<std::string> c(std::vector(1, std::size_t(ntilde)));

        std::stringstream stringify{};
        std::cout << "X = " << X << " Y = " << Y << '\n';
        auto c1_column = xt::view(c, xt::range(0, order));
        for(std::size_t i = 0; i < order; ++i)
        {
            // stringify << " X=" << std::forward<TensorViewX>(X)(i) << "Y=" << std::forward<TensorViewY>(Y)(0) << " = "
            // << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(i), std::forward<TensorViewY>(Y)(0)) <<
            // '\n'
            // ;
            stringify << " X=" << std::forward<TensorViewX>(X)(i) << "Y=" << std::forward<TensorViewY>(Y)(0)
                      << '\n';   // " = " << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(0),
                                 // std::forward<TensorViewY>(Y)(i)) << '\n' ;
            c1_column(i) = stringify.str();
            stringify.str("");
            // matrix_kernel_type{}.evaluate(std::forward<TensorViewY>(Y)(0), std::forward<TensorViewX>(X)(i));
        }

        stringify.str("");
        auto c1_row = xt::view(c, xt::range(order, ntilde));
        for(std::size_t i = 1; i < order; ++i)
        {
            stringify << " X=" << std::forward<TensorViewX>(X)(0) << "Y=" << std::forward<TensorViewY>(Y)(order - i)
                      << '\n';   // " = " << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(order-i),
                                 // std::forward<TensorViewY>(Y)(0)) << '\n';
            // stringify << " X=" << std::forward<TensorViewX>(X)(0) << "Y=" << std::forward<TensorViewY>(Y)(order-i) <<
            // " = " << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(0),
            // std::forward<TensorViewY>(Y)(order-i)) << '\n';
            c1_row(i - 1) = stringify.str();
            stringify.str("");
            // matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(0), std::forward<TensorViewY>(Y)(i));
        }
        return c;
    }
};

template<typename T>
struct TD;
using namespace scalfmm::colors;

template<typename T>
class stdComplex : public std::complex<T>
{
  public:
    using Base = std::complex<T>;
    using Base::Base;

    /** Mul other and another and add the result to current complexe


    */
    /*!
     * \brief addMul perform  z += other*another without using the  function __muldc3 from libgcc. ;
     *
     * "Without -ffast-math or other, complex multiplication yields a call to the function __muldc3 from libgcc. "
     * @see https://stackoverflow.com/questions/42659668/stdcomplex-multiplication-is-extremely-slow
     *
     * \param other
     * \param another
     */
    void addMul(const stdComplex<T>& other, const stdComplex<T>& another)
    {
        //   this->complex[0] += (other.complex[0] * another.complex[0]) - (other.complex[1] * another.complex[1]);
        //     this->complex[1] += (other.complex[0] * another.complex[1]) + (other.complex[1] * another.complex[0]);

        T realPart = this->real() + (other.real() * another.real()) - (other.imag() * another.imag());
        T imagPart = this->imag() + (other.real() * another.imag()) + (other.imag() * another.real());
        this->real(realPart);
        this->imag(imagPart);
    }
};

int main(int argc, char** argv)
{
    static constexpr std::size_t dimension = 3;
    const std::size_t order{3};
    static constexpr std::size_t nnodes = scalfmm::math::pow(order, dimension);
    using value_type = double;

    std::tuple<double, double, double> a{1., 2., 3.};
    std::tuple<double, double, double> b{1., 2., 3.};
    std::tuple<double, double, double> c{1., 2., 3.};

    meta::repeat([](auto& a_, auto const& b_, auto const& c_) { a_ += b_ + c_; }, a, b, c);

    // xt::xarray<scalfmm::container::point<value_type, dimension>> X_points =
    // xt::arange<double>(0,nnodes-1).reshape(std::vector(dimension, order));
    // xt::xarray<scalfmm::container::point<value_type, dimension>> Y_points =
    // xt::arange<double>(0,nnodes-1).reshape(std::vector(dimension, order));

    // build<scalfmm::matrix_kernels::one_over_r, value_type, dimension, dimension> build_c{};
    // std::cout << X_points << '\n';
    // std::cout << Y_points << '\n';

    // std::cout << build_c(X_points, Y_points, order) << '\n';

    // xt::xarray<std::complex<double>> a = xt::ones<std::complex<double>>(std::vector{7,13,13});
    // xt::xarray<std::complex<double>> b = xt::ones<std::complex<double>>(std::vector{7,13,13});
    // xt::xarray<std::complex<double>> c = xt::ones<std::complex<double>>(std::vector{7,13,13});
    // xt::xarray<std::complex<double>> res = xt::zeros<std::complex<double>>(std::vector{7,13,13});

    // std::array<stdComplex<double>, 1183> a_;
    // std::array<stdComplex<double>, 1183> b_;
    // std::array<stdComplex<double>, 1183> c_;
    // stdComplex<double>* a = a_.data();
    // stdComplex<double>* b = b_.data();
    // stdComplex<double>* c = c_.data();

    // xt::xarray<double> a_real = xt::ones<double>(std::vector{7, 13, 13});
    // xt::xarray<double> b_real = xt::ones<double>(std::vector{7, 13, 13});
    // xt::xarray<double> c_real = xt::ones<double>(std::vector{7, 13, 13});
    // xt::xarray<double> r_real = xt::zeros<double>(std::vector{7, 13, 13});
    // xt::xarray<double> a_imag = xt::ones<double>(std::vector{7, 13, 13});
    // xt::xarray<double> b_imag = xt::ones<double>(std::vector{7, 13, 13});
    // xt::xarray<double> c_imag = xt::ones<double>(std::vector{7, 13, 13});
    // xt::xarray<double> r_imag = xt::zeros<double>(std::vector{7, 13, 13});

    // scalfmm::utils::timer<std::chrono::milliseconds> t{};
    // auto size = a.size();
    // static constexpr auto inc = xt_simd::simd_traits<double>::size;
    // auto vec_size = size - size % inc;
    // t.tic();
    // for(std::size_t i = 0; i < 2000000; ++i)
    //{
    // xt::noalias(res) += a*b*double(3.);

    // std::size_t idx{0};
    // std::for_each(std::begin(a), std::end(a),
    //        [&b,&c,&idx](auto& p)
    //
    // for(std::size_t idx = 0; idx < 1099; ++idx)
    //{
    //    a[idx].addMul(stdComplex<double>(8. * c[idx].real(), 8. * c[idx].imag()), b[idx]);
    //    // std::complex tmp(std::real(c.data()[idx])*8., std::imag(c.data()[idx])*8.);
    //    // double ri = std::real(a.data()[idx]) + std::real(tmp)*std::real(b.data()[idx]) -
    //    // std::imag(tmp)*std::imag(b.data()[idx]); double ii = std::real(a.data()[idx]) +
    //    // std::real(tmp)*std::imag(b.data()[idx]) + std::imag(tmp)*std::real(b.data()[idx]);
    //    // a.data()[idx].real(ri); a.data()[idx].imag(ii);
    //    //++idx;
    //}
    //);
    // for(std::size_t j=0; j<size; ++j)
    //{
    //    std::imag(a)
    //}
    // for(std::size_t j=0; j<vec_size; j+=inc)
    //{
    //    auto p_real = a_real.load_simd<xt_simd::aligned_mode, double, inc>(j);
    //    auto p_imag = a_imag.load_simd<xt_simd::aligned_mode, double, inc>(j);
    //    auto m_real = b_real.load_simd<xt_simd::aligned_mode, double, inc>(j);
    //    auto m_imag = b_imag.load_simd<xt_simd::aligned_mode, double, inc>(j);
    //    auto k_real = b_real.load_simd<xt_simd::aligned_mode, double, inc>(j);
    //    auto k_imag = b_imag.load_simd<xt_simd::aligned_mode, double, inc>(j);
    //    auto tmp_real = k_real*8.;
    //    auto tmp_imag = k_imag*8.;
    //    auto real_part = p_real + m_real*k_real - m_imag*k_imag;
    //    auto imag_part = p_imag + m_real*k_imag + m_imag*k_real;
    //    store_aligned(std::addressof(r_real.data()[j]), real_part);
    //    store_aligned(std::addressof(r_real.data()[j]), imag_part);
    //}
    //}
    // std::cout << t.tac_and_elapsed() << '\n';
    // std::cout << res << '\n';
    // xt::xarray<xtl::xcomplex<double>> b = xt::arange<xtl::xcomplex<double>>(0, 8).reshape(std::vector(3,2));
    // xt::xarray<xtl::xcomplex<double>> c = xt::arange<xtl::xcomplex<double>>(0, 8).reshape(std::vector(3,2));

    // using batch_type = xsimd::simd_type<xtl::xcomplex<double>>;
    // using batch_type = xsimd::simd_type<xtl::xcomplex<double>>;

    // auto a = batch_type(xtl::xcomplex<double>(1.,1.));
    // auto b = batch_type(xtl::xcomplex<double>(1.,1.));
    // auto c = batch_type(xtl::xcomplex<double>(1.,1.));

    // auto tmp = xsimd::fma(a, b, c);
    // auto tmp = xsimd::sqrt(a);
    // auto tmp = xt::fma(a, b, c);

    // std::cout << tmp << '\n';

    return 0;
}
