#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <xtensor/xarray.hpp>

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/utils/tensor.hpp"
#include <scalfmm/container/point.hpp>
#include <scalfmm/interpolation/mapping.hpp>
#include <scalfmm/utils/math.hpp>

// using namespace scalfmm;
// template<typename MatrixKernel, typename ValueType, std::size_t Dimension, std::size_t SavedDimension>
// struct build
// {
//     using matrix_kernel_type = MatrixKernel;
//     using value_type = ValueType;

//     template<typename TensorViewX, typename TensorViewY>
//     [[nodiscard]] inline auto operator()(TensorViewX&& X, TensorViewY&& Y, std::size_t order)
//     {
//         static constexpr std::size_t current_dimension = Dimension;
//         static constexpr std::size_t saved_dimension = SavedDimension;
//         static constexpr std::size_t dimension_decrease = current_dimension - 1;
//         std::size_t ntilde = (2 * order) - 1;
//         xt::xarray<std::string> c(std::vector(current_dimension, std::size_t(ntilde)));
//         build<matrix_kernel_type, value_type, dimension_decrease, saved_dimension> build_c1{};

//         // auto range = xt::all();
//         // for(std::size_t i = 0; i<order; ++i)
//         //{
//         //    auto c1 = tensor::get_view<dimension_decrease>(c, i, range, tensor::row{});
//         //    auto X_view =
//         //      tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), i, range, tensor::column{});
//         //    auto Y_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), 0, range,
//         tensor::row{});
//         //    c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order);
//         //}
//         for(std::size_t i = 0; i < order; ++i)
//         {
//             auto range = xt::all();
//             auto c1 = tensor::get_view<dimension_decrease>(c, i, range, tensor::row{});
//             auto X_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), i, range,
//             tensor::row{});
//             // auto X_view = tensor::gather<current_dimension>(X, i);
//             std::cout << "X_view=" << X_view << '\n';
//             // auto Y_view = tensor::gather<current_dimension>(Y, 0);
//             auto Y_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), 0, range,
//             tensor::row{});
//             // auto Y_view = xt::view(Y, xt::all(), xt::keep(0));
//             // std::cout << "Y_view=" << Y_view << '\n';
//             // tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), 0, range, tensor::row{});
//             c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order);
//         }

//         for(std::size_t i = 1; i < order; ++i)
//         {
//             auto range = xt::all();
//             auto c1 = tensor::get_view<dimension_decrease>(c, order - 1 + i, range, tensor::row{});
//             auto X_view = tensor::get_view<dimension_decrease>(std::forward<TensorViewX>(X), 0, range,
//             tensor::row{});
//             // auto X_view = tensor::gather<current_dimension>(X, 0);
//             std::cout << "X_view=" << X_view << '\n';
//             // auto Y_view = tensor::gather<current_dimension>(Y, order-i);
//             auto Y_view =
//               tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), order - i, range, tensor::row{});
//             // auto Y_view = xt::view(Y, xt::all(), xt::keep(order-i));
//             std::cout << "Y_view=" << Y_view << '\n';
//             // tensor::get_view<dimension_decrease>(std::forward<TensorViewY>(Y), order-i, range, tensor::row{});
//             // //column
//             c1 = build_c1(std::forward<decltype(X_view)>(X_view), std::forward<decltype(Y_view)>(Y_view), order);
//         }
//         return c;
//     }
// };

// template<typename MatrixKernel, typename ValueType, std::size_t SavedDimension>
// struct build<MatrixKernel, ValueType, 1, SavedDimension>
// {
//     using matrix_kernel_type = MatrixKernel;
//     using value_type = ValueType;

//     template<typename TensorViewX, typename TensorViewY>
//     [[nodiscard]] inline auto operator()(TensorViewX&& X, TensorViewY&& Y, std::size_t order)
//     {
//         std::size_t ntilde = 2 * order - 1;
//         xt::xarray<std::string> c(std::vector(1, std::size_t(ntilde)));

//         std::stringstream stringify{};
//         std::cout << "X = " << X << " Y = " << Y << '\n';
//         auto c1_column = xt::view(c, xt::range(0, order));
//         for(std::size_t i = 0; i < order; ++i)
//         {
//             // stringify << " X=" << std::forward<TensorViewX>(X)(i) << "Y=" << std::forward<TensorViewY>(Y)(0) << "
//             = "
//             // << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(i), std::forward<TensorViewY>(Y)(0)) <<
//             // '\n'
//             // ;
//             stringify << " X=" << std::forward<TensorViewX>(X)(i) << "Y=" << std::forward<TensorViewY>(Y)(0)
//                       << '\n';   // " = " << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(0),
//                                  // std::forward<TensorViewY>(Y)(i)) << '\n' ;
//             c1_column(i) = stringify.str();
//             stringify.str("");
//             // matrix_kernel_type{}.evaluate(std::forward<TensorViewY>(Y)(0), std::forward<TensorViewX>(X)(i));
//         }

//         stringify.str("");
//         auto c1_row = xt::view(c, xt::range(order, ntilde));
//         for(std::size_t i = 1; i < order; ++i)
//         {
//             stringify << " X=" << std::forward<TensorViewX>(X)(0) << "Y=" << std::forward<TensorViewY>(Y)(order - i)
//                       << '\n';   // " = " << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(order-i),
//                                  // std::forward<TensorViewY>(Y)(0)) << '\n';
//             // stringify << " X=" << std::forward<TensorViewX>(X)(0) << "Y=" << std::forward<TensorViewY>(Y)(order-i)
//             <<
//             // " = " << matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(0),
//             // std::forward<TensorViewY>(Y)(order-i)) << '\n';
//             c1_row(i - 1) = stringify.str();
//             stringify.str("");
//             // matrix_kernel_type{}.evaluate(std::forward<TensorViewX>(X)(0), std::forward<TensorViewY>(Y)(i));
//         }
//         return c;
//     }
// };

int main(int argc, char** argv)
{
    static constexpr std::size_t dimension = 2;

    using value_type = double;
    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    using interpolator_type =
      scalfmm::interpolation::interpolator<value_type, dimension, matrix_kernel_type,
                                           scalfmm::options::chebyshev_<scalfmm::options::dense_>>;
    const std::size_t order{4};
    static constexpr std::size_t nnodes = scalfmm::math::pow(order, dimension);
    std::size_t tree_height{3};
    value_type width{7.0};

    interpolator_type interpolator(matrix_kernel_type{}, order, tree_height, width);
    // roots_1d in cell [-0.5,0.5]
    auto roots_1d = 0.5 * interpolator.roots();
    std::cout << "roots: " << roots_1d << std::endl;
    scalfmm::container::point<double, dimension> center(0.0);

    auto roots = scalfmm::tensor::generate_meshgrid<dimension>(roots_1d);
    std::cout << "roots\n" << std::get<0>(roots) << std::endl;
    //  const auto roots_x = xt::flatten(std::get<0>(roots));
    // const auto roots_y = xt::flatten(std::get<1>(roots));
    // const auto roots_z = xt::flatten(std::get<2>(roots));
    // // Set the collocation points inside the cell
    // value_type half_width = 0.5;

    // auto pos_x = half_width * roots_x + center[0];
    // auto pos_y = half_width * roots_y + center[1];
    // auto pos_z = half_width * roots_z + center[2];

    xt::xarray<double> Id = xt::eye<double>({nnodes, nnodes});
    xt::xarray<double> A;
    std::vector shape{nnodes, 2};
    // static constexpr xt::layout_type col = xt::layout_type::column_major;
    // static constexpr xt::layout_type row = xt::layout_type::row_major;
    xt::xarray<double, xt::layout_type::row_major> B = xt::arange<double>(0, 2 * nnodes).reshape(shape);
    xt::xarray<double, xt::layout_type::row_major> C = xt::zeros<double>(shape);
    // xt::arange<double>(0,nnodes-1).reshape(std::vector(dimension, order));
    // xt::xarray<scalfmm::container::point<value_type, dimension>> Y_points =
    // xt::arange<double>(0,nnodes-1).reshape(std::vector(dimension, order));

    //   build<scalfmm::matrix_kernels::one_over_r, value_type, dimension, dimension> build_c{};
    std::cout << "B " << B.data() << "\n" << B << std::endl;
    scalfmm::tensor::blas3_product(B, C, Id, 2, 1.0, false);
    std::cout << "C \n" << C << std::endl;

    // build_c << '\n';
    // // std::cout << Y_points << '\n';

    return 0;
}
