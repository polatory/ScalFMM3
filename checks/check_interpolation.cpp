#include <array>
#include <random>
#include <typeinfo>
#include <vector>
#include <xsimd/xsimd.hpp>

#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "xsimd/xsimd.hpp"

#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

using value_type = double;
static constexpr std::size_t dimension{3};
using namespace scalfmm;

namespace local_args
{
    struct number_of_points
    {
        cpp_tools::cl_parser::str_vec flags = {"--number-of-points", "--N"};
        std::string description = "The number of points at which the polynomials are evaluated.";
        using type = int;
        type def = 100;
    };
}   // namespace local_args

// Function that checks an interpolator
template<typename InterpolatorType>
void process_interpolator(const int& N, const int& order)
{
    using value_type = typename InterpolatorType::value_type;
    using simd_type = xsimd::simd_type<value_type>;
    constexpr std::size_t simd_size = simd_type::size;

    cpp_tools::timers::timer<std::chrono::nanoseconds> time{};

    std::size_t tree_height{3};
    value_type root_cell_width{1.};

    InterpolatorType interp(order, tree_height, root_cell_width);
    xt::xarray<value_type> roots = interp.roots_impl();
    std::size_t vec_size = order - order % simd_size;

    std::cout << std::fixed << std::endl;

    // print roots
    std::cout << cpp_tools::colors::green;
    std::cout << "roots: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    std::cout << std::showpos;
    for(const auto& r: roots)
    {
        std::cout << r << " ";
    }
    std::cout << std::noshowpos << std::endl;

    // evaluate each polynomial at each root
    std::cout << cpp_tools::colors::green;
    std::cout << "P_o(roots): " << std::endl;
    std::cout << cpp_tools::colors::reset;
    std::cout << std::showpos;
    for(unsigned int o = 0; o < order; ++o)
    {
        for(const auto& r: roots)
        {
            std::cout << interp.polynomials_impl(r, o) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::noshowpos;

    // evaluate each polynomial at each root (using simd)
    std::cout << cpp_tools::colors::green;
    std::cout << "P_o(roots) [simd]: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    std::cout << std::showpos;
    for(unsigned int o = 0; o < order; ++o)
    {
        for(std::size_t k = 0; k < vec_size; k += simd_size)
        {
            simd_type roots_simd = xsimd::load_aligned(&roots[k]);
            std::cout << interp.polynomials_impl(roots_simd, o) << " ";
        }
        for(std::size_t k = vec_size; k < order; ++k)
        {
            std::cout << interp.polynomials_impl(roots[k], o) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::noshowpos;

    // evaluate each derivative at each root
    std::cout << cpp_tools::colors::green;
    std::cout << "dP_o(roots)/dx: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    std::cout << std::showpos;
    for(unsigned int o = 0; o < order; ++o)
    {
        for(const auto& r: roots)
        {
            std::cout << interp.derivative_impl(r, o) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::noshowpos;

    // evaluate each derivative at each root (using simd)
    std::cout << cpp_tools::colors::green;
    std::cout << "dP_o(roots)/dx [simd]: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    std::cout << std::showpos;
    for(unsigned int o = 0; o < order; ++o)
    {
        for(std::size_t k = 0; k < vec_size; k += simd_size)
        {
            simd_type roots_simd = xsimd::load_aligned(&roots[k]);
            std::cout << interp.derivative_impl(roots_simd, o) << " ";
        }
        for(std::size_t k = vec_size; k < order; ++k)
        {
            std::cout << interp.derivative_impl(roots[k], o) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::noshowpos;

    // check wether the sums of the derivatives vanish
    std::cout << cpp_tools::colors::green;
    std::cout << "sums: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    value_type current_sum{}, current_root{};
    std::cout << std::showpos;
    for(unsigned int o = 0; o < order; ++o)
    {
        current_sum = 0.;
        current_root = roots[o];
        for(unsigned int k = 0; k < order; ++k)
        {
            current_sum += interp.derivative_impl(current_root, k);
        }
        std::cout << "o = " << o << ": " << current_sum << std::endl;
    }
    std::cout << std::noshowpos;

    // evaluation at random points (not necessarily roots)
    std::cout << cpp_tools::colors::green;
    std::cout << "random evaluations: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    std::array<value_type, 3> values{-0.12748145601659977, 0.4579302226486055, 0.24907712585461272};

    std::cout << std::showpos;
    for(unsigned int o = 0; o < order; ++o)
    {
        for(const auto& v: values)
        {
            std::cout << "P_" << o << "(" << v << ") = " << interp.polynomials_impl(v, o) << "\t";
        }
        std::cout << std::endl;
    }
    for(unsigned int o = 0; o < order; ++o)
    {
        for(const auto& v: values)
        {
            std::cout << "dP_" << o << "(" << v << ")/dx = " << interp.derivative_impl(v, o) << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::noshowpos;

    // assess computation time
    std::vector<value_type> points(N, 0.0);
    const auto seed{0};
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<> dist(-1, 1);
    for(auto& v: points)
    {
        v = dist(gen);
    }

    std::cout << cpp_tools::colors::green;
    std::cout << "polynomials: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    value_type val{};
    time.tic();
    for(auto v: points)
    {
        val = interp.polynomials_impl(v, order - 1);
    }
    time.tac();
    std::cout << "- elapsed time (over " << N << " iterations): " << time.elapsed() << " ns" << std::endl;

    std::cout << cpp_tools::colors::green;
    std::cout << "derivative: " << std::endl;
    std::cout << cpp_tools::colors::reset;
    time.tic();
    for(auto v: points)
    {
        val = interp.derivative_impl(v, order - 1);
    }
    time.tac();
    std::cout << "- elapsed time (over " << N << " iterations): " << time.elapsed() << " ns" << std::endl;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::order{}, local_args::number_of_points{});
    parser.parse(argc, argv);

    // Getting command line parameters
    const int N{parser.get<local_args::number_of_points>()};
    std::cout << cpp_tools::colors::blue << "<params> Number of points : " << N << cpp_tools::colors::reset << '\n';
    const auto order{parser.get<args::order>()};
    std::cout << cpp_tools::colors::blue << "<params> order : " << order << " (degree = " << (order - 1) << ")"
              << cpp_tools::colors::reset << '\n';

    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;

    // Interpolator aliases
    using interpolator_cheb_type =
      interpolation::chebyshev_interpolator<value_type, dimension, matrix_kernel_type, scalfmm::options::low_rank_>;
    using interpolator_unif_type =
      interpolation::uniform_interpolator<value_type, dimension, matrix_kernel_type, scalfmm::options::low_rank_>;
    using interpolator_barycentric_type =
      interpolation::barycentric_interpolator<value_type, dimension, matrix_kernel_type, scalfmm::options::low_rank_>;
    using interpolator_unif_fft_type =
      interpolation::uniform_interpolator<value_type, dimension, matrix_kernel_type, scalfmm::options::fft_>;

    // Compare interpolators
    std::cout << cpp_tools::colors::cyan << "\n--- CHEBYSHEV ---" << std::endl;
    process_interpolator<interpolator_cheb_type>(N, order);

    std::cout << cpp_tools::colors::cyan << "\n--- BARYCENTRIC ---" << std::endl;
    process_interpolator<interpolator_barycentric_type>(N, order);

    std::cout << cpp_tools::colors::cyan << "\n--- UNIFORM (fft) ---" << std::endl;
    process_interpolator<interpolator_unif_fft_type>(N, order);

    std::cout << cpp_tools::colors::cyan << "\n--- UNIFORM ---" << std::endl;
    process_interpolator<interpolator_unif_type>(N, order);

    return 0;
}
