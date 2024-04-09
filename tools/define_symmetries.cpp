/*
 * genarateDistributions.cpp
 *
 *  Created on: 23 mars 2014
 *      Author: Olivier Coulaud
 */
// @FUSE_CBLAS
// @FUSE_FFTW
#include <cstdlib>
//#include <fstream>
#include <iostream>
#include <vector>
//
#include <xtensor/xarray.hpp>
// #include <xtensor/xeval.hpp>
// #include <xtensor/xmanipulation.hpp>
//
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/utils/tensor.hpp"
#include <scalfmm/meta/utils.hpp>
#include <scalfmm/utils/io_helpers.hpp>
#include <scalfmm/utils/math.hpp>
//#include "scalfmm/interpolation/uniform.hpp"
#include "scalfmm/tree/box.hpp"
#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>

using value_type = double;

using namespace scalfmm::out;
// template<typename T, std::size_t N>
// inline auto operator<<(std::ostream& os, const std::array<T, N>& array) -> std::ostream&
// {
//     os << "[";
//     for(auto it = array.begin(); it != array.end() - 1; it++)
//     {
//         os << *it << ", ";
//     }
//     os << array.back() << "]";
//     return os;
// }
namespace local_args
{

    struct dimensionSpace
    {
        cpp_tools::cl_parser::str_vec flags = {"--dim", "--dimension", "--d"};
        std::string description = "Dimension of the space (d < 5)";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
        type def = 3;
    };

}   // namespace local_args
template<typename INDEX, typename INT>
auto full_index_col(INDEX const& index, INT const& N) -> typename INDEX::value_type
{
    constexpr static std::size_t dimension = std::tuple_size<INDEX>::value;
    if constexpr(dimension == 2)
    {
        return index[0] * N + index[1];
    }
    else if constexpr(dimension == 3)
    {
        return (index[0] * N + index[1]) * N + index[2];
    }
    else
    {
        throw("Dimension non supported (only 2 & 3");
    }
}
template<typename INDEX, typename INT>
auto full_index_row(INDEX const& index, INT const& N) -> typename INDEX::value_type
{
    constexpr static std::size_t dimension = std::tuple_size<INDEX>::value;

    if constexpr(dimension == 2)
    {
        return index[1] * N + index[0];
    }
    else if constexpr(dimension == 3)
    {
        return (index[2] * N + index[1]) * N + index[0];
    }
    else
    {
        throw("Dimension non supported (only 2 & 3");
    }
}
template<typename Point, typename Int>
auto diag_index(Point const& t, Int& axe) -> Point
{
    static constexpr std::size_t Dimension = std::tuple_size_v<Point>;
    if constexpr(Dimension == 2)
    {
        return Point{t[1], t[0]};
    }
    else if constexpr(Dimension == 3)
    {
        if(axe == 0)
        {
            return Point{t[0], t[2], t[1]};
        }
        else if(axe == 1)
        {
            return Point{t[2], t[1], t[0]};
        }
        else if(axe == 2)
        {
            return Point{t[1], t[0], t[2]};
        }
        else
        {
            throw("wrong axe!");
            return Point{};
        }

    }
    else
    {
        throw("Dimension not yet implemented in transfert_vector");
        return Point{};
    }
}
// template<int... IS>
// auto index_row_major(int const& N, std::index_sequence<IS...> is) -> void
// {
//     static constexpr std::size_t dimension = sizeof(std::index_sequence<IS...>);
//     std::array<int, dimension> index{is};
//     ifconstexpr(dimension == 2) { return index[1] * N + index[0]; }
//     else ifconstexpr(dimension == 2) { return (index[2] * N + index[1]) * N + index[0]; }
//     else { throw("Dimension non supported"); }
// }
template<int Dimension, typename MatrixKernel_type, typename Grid>
auto check_type1_symmetries(MatrixKernel_type& mat, Grid const& X, scalfmm::container::point<value_type, Dimension> t,
                            const int& order) -> bool
{
    using matrix_type = typename MatrixKernel_type::template matrix_type<value_type>;
    constexpr static std::size_t nvals = std::tuple_size<matrix_type>::value;
    constexpr value_type eps = 1.0e-6;
    std::cout << cpp_tools::colors::blue << "Entering check symmetries type 1 ...\n" << cpp_tools::colors::reset;

    //
    // First translated grid
    auto Y = t + X;
    std::vector<std::array<int, Dimension>> ijk(X.size()), ijkp(X.size());
    std::array<int, Dimension> start, stop;
    start.fill(0);
    stop.fill(order);
    int pos = 0;
    auto fill_ijk = [&ijk, &pos](auto... is)
    {
        ijk[pos] = {is...};
        ++pos;
    };
    // construct the ijk indexes associated to l
    scalfmm::meta::looper_range<Dimension>{}(fill_ijk, start, stop);
    // eval the kernel on both grid with x in X
    //  K1 = kernel(X,Y)  and  K2 = kernel(X,Yt)
    std::vector<size_t> shape = {X.size(), Y.size()};
    xt::xarray<matrix_type> K1(shape), K2(shape);

    for(int l{0}; l < X.size(); ++l)
    {
        for(int k{0}; k < X.size(); ++k)
        {
            K1.at(l, k) = mat.evaluate(X[l], Y[k]);
        }
    }   //
    // Check axial symmetries
    bool Ok = true;
    auto nb_symmetries = Dimension;
    for(int i{0}; i < nb_symmetries; ++i)
    {
        //   std::cout << "   Axe symmetries " << i;
        auto tp{t};
        tp[i] *= -1.0;
        //  std::cout << "  new vector " << tp << std::endl;
        //
        // std::copy(ijk.begin(), ijk.end(), ijkp.begin());
        std::transform(ijk.begin(), ijk.end(), ijkp.begin(),
                       [order, i](auto c)
                       {
                           auto a = c;
                           a[i] = order - 1 - a[i];
                           return a;
                       });
        int pos = 0;

        auto Yt = tp + X;
        // eval the kernel on both grid with x in X
        //  F1 = kernel(X,Y)  and  F2 = kernel(X,Yt)
        for(int l{0}; l < X.size(); ++l)
        {
            for(int k{0}; k < X.size(); ++k)
            {
                K2.at(l, k) = mat.evaluate(X[l], Yt[k]);
            }
        }
        // Check axial symmetry
        for(int l{0}; l < X.size(); ++l)
        {
            auto nl = full_index_col(ijkp[l], order);

            for(int k{0}; k < X.size(); ++k)
            {
                auto nk = full_index_col(ijkp[k], order);

                for(int m{0}; m < nvals; ++m)
                {
                    Ok = Ok && (std::abs(K1.at(l, k)[m] - K2.at(nl, nk)[m]) < eps);
                }
                if(!Ok)
                {
                    std::cout << l << " " << k << " " << K1.at(l, k) << " " << nl << " " << nk << " " << K2[nl, nk]
                              << std::endl;
                    for(int m{0}; m < nvals; ++m)
                    {
                        std::cout << "       " << m << " diff( " << K1.at(l, k)[m] << " - " << K2.at(nl, nk)[m]
                                  << " ) = " << std::abs(K1.at(l, k)[m] - K2.at(nl, nk)[m]) << std::endl;
                    }

                    std::cerr << "     Symmetry type 1 on axis " << i << " No" << std::endl;
                    goto end_loop1;
                }
            }
        }
    end_loop1:

        if(Ok)
        {
            std::cout << "     Symmetry type 1 on axis " << i << " Yes" << std::endl;
        }
        // else
        // {
        //     break;
        // }
    }
    return Ok;
}
template<typename Point>
auto diag_vector(Point const& t, int& axe) -> Point
{
    static constexpr std::size_t Dimension = Point::dimension;
    if constexpr(Dimension == 2)
    {
        return Point({t[1], t[0]});
    }
    else if constexpr(Dimension == 3)
    {
        if(axe == 0)
        {
            return Point({t[0], t[2], t[1]});
        }
        else if(axe == 1)
        {
            std::cout << " axe " << axe << " Point " << Point({t[2], t[1], t[0]}) << std::endl;
            return Point({t[2], t[1], t[0]});
        }
        else if(axe == 2)
        {
            return Point({t[1], t[0], t[2]});
        }
        else
        {
            throw("wrong axe!");
            return Point{};
        }
    }
    else
    {
        throw("Dimension not yet implemented in transfert_vector");
        return Point{};
    }
}
template<int Dimension, typename value_type>
auto transfert_vector() -> scalfmm::container::point<value_type, Dimension>
{
    if constexpr(Dimension == 2)
    {
        return scalfmm::container::point<value_type, Dimension>{2, 1};
    }
    else if constexpr(Dimension == 3)
    {
        return scalfmm::container::point<value_type, Dimension>{2, 2, 1};
    }
    else
    {
        throw("Dimension not yet implemented in transfert_vector");
        return scalfmm::container::point<value_type, Dimension>{};
    }
}
template<int Dimension, typename MatrixKernel_type, typename Grid>
auto check_type2_symmetries(MatrixKernel_type& mat, Grid const& X, scalfmm::container::point<value_type, Dimension> t,
                            const int& order) -> bool
{
    using matrix_type = typename MatrixKernel_type::template matrix_type<value_type>;
    constexpr static std::size_t nvals = std::tuple_size<matrix_type>::value;
    constexpr value_type eps = 1.0e-15;
    std::cout << cpp_tools::colors::blue << "Entering check symmetries type 2...\n" << cpp_tools::colors::reset;
    //
    // First translated grid
    auto Y = t + X;
    std::vector<std::array<int, Dimension>> ijk(X.size()), ijkp(X.size());
    std::array<int, Dimension> start, stop;
    start.fill(0);
    stop.fill(order);
    int pos = 0;
    auto fill_ijk = [&ijk, &pos](auto... is)
    {
        ijk[pos] = {is...};
        ++pos;
    };
    // construct the ijk indexes associated to l
    scalfmm::meta::looper_range<Dimension>{}(fill_ijk, start, stop);
    // eval the kernel on both grid with x in X
    //  K1 = kernel(X,Y)  and  K2 = kernel(X,Yt)
    std::vector<size_t> shape = {X.size(), Y.size()};
    xt::xarray<matrix_type> K1(shape), K2(shape);

    for(int l{0}; l < X.size(); ++l)
    {
        for(int k{0}; k < X.size(); ++k)
        {
            K1.at(l, k) = mat.evaluate(X[l], Y[k]);
        }
    }   //
    // Check axial symmetries

    bool Ok = true;
    int nb_symmetries =
      int(scalfmm::math::factorial<int>(Dimension) / (scalfmm::math::factorial<int>(Dimension - 2) * 2));
    std::cout << "Number of symmetries of type 2 " << nb_symmetries << std::endl;
    for(int axe{0}; axe < nb_symmetries; ++axe)
    {
        // std::cout << "   Diag symmetries " << axe;

        auto tp = diag_vector(t, axe);
        // std::cout << "  new vector " << tp << std::endl;
        //
        // std::copy(ijk.begin(), ijk.end(), ijkp.begin());
        std::transform(ijk.begin(), ijk.end(), ijkp.begin(), [axe](auto c) { return diag_index(c, axe); });
        // for(int l{0}; l < X.size(); ++l)
        // {
        //     std::cout << l << "=" << ijk[l] << "  perm " << ijkp[l] << std::endl;
        // }
        int pos = 0;
        auto Yt = tp + X;
        // eval the kernel on both grid with x in X
        //  F1 = kernel(X,Y)  and  F2 = kernel(X,Yt)
        for(int l{0}; l < X.size(); ++l)
        {
            for(int k{0}; k < X.size(); ++k)
            {
                K2.at(l, k) = mat.evaluate(X[l], Yt[k]);
            }
        }
        // Check axial symmetry
        for(int l{0}; l < X.size(); ++l)
        {
            auto nl = full_index_col(ijkp[l], order);

            for(int k{0}; k < X.size(); ++k)
            {
                auto nk = full_index_col(ijkp[k], order);

                for(int m{0}; m < nvals; ++m)
                {
                    Ok = Ok && (std::abs(K1.at(l, k)[m] - K2.at(nl, nk)[m]) < eps);

                    if(!Ok)
                    {
                        std::cout << l << " " << k << " " << K1.at(l, k) << " " << nl << " " << nk << " " << K2[nl, nk]
                                  << std::endl;
                        for(int m{0}; m < nvals; ++m)
                        {
                            std::cout << "       " << m << " diff( " << K1.at(l, k)[m] << " - " << K2.at(nl, nk)[m]
                                      << " ) = " << std::abs(K1.at(l, k)[m] - K2.at(nl, nk)[m]) << std::endl;
                        }

                        std::cerr << "     Symmetry type 2  num  " << axe << " No" << std::endl;
                        goto end_loop2;
                    }
                }
            }
        }
    end_loop2:
        if(Ok)
        {
            std::cout << "     Symmetry type 2  num  " << axe << " Yes" << std::endl;
        }
    }
    return Ok;
}

template<int Dimension, typename MatrixKernel_type>
auto check_symmetries(MatrixKernel_type& mat) -> void
{
    //
    using interpolator_type =
      scalfmm::interpolation::interpolator<value_type, Dimension, MatrixKernel_type,
                                           scalfmm::options::chebyshev_<scalfmm::options::dense_>>;

    using matrix_type = typename MatrixKernel_type::template matrix_type<value_type>;
    const std::size_t tree_height{3};
    const value_type box_width{1.};
    const int order = 3;

    // We construct an fmm_operator for the call to the l2p operator.
    // This will allow us to distinguish optimized version of the l2p operator for
    // specific laplace kernels combination.
    interpolator_type interpolator(MatrixKernel_type{}, order, tree_height, box_width);

    // Then the cell
    // cell_type cell(center, width, order);

    auto roots = interpolator.roots();
    std::cout << "roots: ";
    std::cout << roots << std::endl;
    // Here we generate a grid of points corresponding to the roots in the targeted dimension.
    auto X = scalfmm::tensor::generate_grid_of_points<Dimension>(roots);

    bool symm = true;

    // Build the transfert vector
    scalfmm::container::point<value_type, Dimension> t(2);
    t[Dimension - 1] = 1;
    //
    //  check type 1 symmetries
    symm = symm && check_type1_symmetries<Dimension>(mat, X, t, order);
    if(symm)
    {
        std::cout << cpp_tools::colors::red << " Kernel verifies symetries type 1 " << std::endl
                  << cpp_tools::colors::reset;
    }
    //  check diagonal symmetries
    if constexpr(Dimension > 1)
    {
        std::cout << "TEst" << std::endl;
        bool symm2 = check_type2_symmetries<Dimension>(mat, X, t, order);
        symm = symm && symm2;
        std::cout << cpp_tools::colors::red;

        if(symm)
        {
            std::cout << " Kernel verifies symetries type 2 " << std::endl << cpp_tools::colors::reset;
        }
        else
        {
            std::exit(EXIT_FAILURE);
        };
        // if constexpr(Dimension > 2)
        // {
        //     symm = symm && check_type3_symmetries<Dimension>(mat, X, t, order);
        //     std::cout << cpp_tools::colors::red;

        //     if(symm)
        //     {
        //         std::cout << " Kernel verifies symetries type 3 " << std::endl << cpp_tools::colors::reset;
        //     }
        //     else
        //     {
        //         std::exit(EXIT_FAILURE);
        //     };
    }
}
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    /// Parsing options
    ///
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, local_args::dimensionSpace{});

    // Parameter handling
    parser.parse(argc, argv);

    std::cout << cpp_tools::colors::blue << "Entering check_symmetries ...\n" << cpp_tools::colors::reset;

    auto dimension = parser.get<local_args::dimensionSpace>();

    if(dimension == 1)
    {
        using matrix_type = scalfmm::matrix_kernels::laplace::one_over_r;

        matrix_type mk{};
        check_symmetries<1>(mk);
    }
    else if(dimension == 2)
    {
        // using matrix_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<2>;
        using matrix_type = scalfmm::matrix_kernels::laplace::grad_ln_2d;
        // using matrix_type = scalfmm::matrix_kernels::laplace::ln_2d;
        matrix_type mk{};
        check_symmetries<2>(mk);
    }
    else if(dimension == 3)
    {
        using matrix_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<3>;
        matrix_type mk{};
        check_symmetries<3>(mk);
    }
    // else if(dimension == 4)
    // {
    //     change_format<4>(input_file, use_old_format, visu_file, output_file);
    // }
    // else
    // {
    //     std::cerr << "Dimension should be lower than 5 (d<5)" << std::endl;
    //     std::exit(EXIT_FAILURE);
    // }
    return 0;
}
