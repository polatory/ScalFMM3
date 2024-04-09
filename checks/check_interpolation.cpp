#include <random>
#include <vector>

#include <cpp_tools/timers/simple_timer.hpp>

#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"

int main()
{
    int res{0};
    constexpr int dimension{3};
    using value_type = float;
    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;

    // scalfmm::interpolation::chebyshev_interpolator
    using interpolator_cheb_type =
      scalfmm::interpolation::chebyshev_interpolator<value_type, dimension, matrix_kernel_type,
                                                     scalfmm::options::low_rank_>;
    using interpolator_unif_type =
      scalfmm::interpolation::uniform_interpolator<value_type, dimension, matrix_kernel_type,
                                                     scalfmm::options::fft_>;
    using interpolator_unif_dense_type =
      scalfmm::interpolation::uniform_interpolator<value_type, dimension, matrix_kernel_type,
                                                     scalfmm::options::dense_>;matrix_kernel_type mk;
    int order{6};
    int N = 10000;
    std::vector<value_type> points(N, 0.0);
    const auto seed{0};
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<> dist(-1, 1);
    for(auto& v: points)
    {
        v = dist(gen);
    }

    cpp_tools::timers::timer<std::chrono::nanoseconds> time{};

    interpolator_cheb_type inter_cheb(mk, order);
    interpolator_unif_type inter_unif(mk, order);
    interpolator_unif_dense_type inter_unif_dense(mk, order);

    value_type val{};
    time.tic();
    for(auto v: points)
    {
        val = inter_cheb.polynomials_impl(v, order - 1);
    }
    time.tac();
    std::cout << " cheb polynomials_impl   in " << time.elapsed() << " ms" << std::endl;
    std::cout << " val " << val << std::endl;

    time.tic();
    for(auto v: points)
    {
        val = inter_cheb.derivative_impl(v, order - 1);
    }
    time.tac();
    std::cout << " cheb derivative_impl  in " << time.elapsed() << " ms" << std::endl;
    std::cout << " val " << val << std::endl;
    //////////////////////////////////////////
    // Uniform
    time.tic();
    for(auto v: points)
    {
        val = inter_unif_dense.polynomials_impl(v, order - 1);
    }
    time.tac();
    auto timeold = time.elapsed();
    std::cout << " dense unif polynomials_impl   in " << timeold << " ms" << std::endl;
    std::cout << " val " << val << std::endl;

    time.tic();
    for(auto v: points)
    {
        val = inter_unif.polynomials_impl(v, order - 1);
    }
    time.tac();
    auto timenew = time.elapsed();
    std::cout << " fft unif polynomials_impl   in " << timenew << " ms"<< " ratio " <<double(timeold)/double(timenew) << std::endl;
    std::cout << " val " << val << std::endl;

    time.tic();
    for(auto v: points)
    {
        val = inter_unif.derivative_impl1(v, order - 1);
    }
    time.tac();
    timeold = time.elapsed();
    std::cout << "old unif derivative_impl  in " << timeold << " ms" << std::endl;
    std::cout << " val " << val << std::endl;

    time.tic();
    for(auto v: points)
    {
        val = inter_unif.derivative_impl(v, order - 1);
    }
    time.tac();
     timenew = time.elapsed() ;
    std::cout << " unif derivative_impl  in " << time.elapsed() << " ms" << " ratio " <<double(timeold)/double(timenew)<< std::endl;
    std::cout << " val " << val << std::endl;
    /////////////////
    std::cout << " Check function\n";
    auto  val1 = inter_unif.polynomials_impl(points[10], order - 1);
    auto  val2 = inter_unif_dense.polynomials_impl(points[10], order - 1);
    auto  val3 = inter_cheb.polynomials_impl(points[10], order - 1);
    std::cout << " val(unif_opt)" << val1 << " val(inf_dense) " << val2<<  " val(cheb) " << val3 << std::endl;
    std::cout << " Check derivative\n";
      auto val4 = inter_unif.derivative_impl(points[10], order - 1);
    auto  val5 = inter_unif_dense.derivative_impl(points[10], order - 1);
    auto  val6 = inter_cheb.derivative_impl(points[10], order - 1);
    std::cout << " val(unif_opt) " << val4 << " val(inf_dense) " << val5<<  " val(cheb) " << val6 << std::endl;

    return res;
}