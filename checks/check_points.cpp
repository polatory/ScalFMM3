/**
 * @file check_points.cpp
 * @author Antoine Gicquel
 * @brief This file checks the equality operator of the points.
 *
 * @date 2025-06-01
 *
 */

#include "scalfmm/container/point.hpp"

template<typename ValueType, std::size_t Dimension>
auto run_equality_point() -> void
{
    using value_type = ValueType;
    static constexpr std::size_t dimension = Dimension;

    std::cout << std::boolalpha;

    // Test equality operator with classical points
    using point_impl_type = scalfmm::container::point<value_type, dimension>;

    point_impl_type p1(1);
    point_impl_type p2(1);
    point_impl_type p3(2);

    std::cout << "- p1 = " << p1 << std::endl;
    std::cout << "- p2 = " << p2 << std::endl;
    std::cout << "- p3 = " << p3 << std::endl;
    std::cout << "- p1 == p2 ? : " << (p1 == p2) << std::endl;
    std::cout << "- p1 == p3 ? : " << (p1 == p3) << std::endl;
    std::cout << std::endl;

    // Test equality operator with proxy points (points of references)
    using proxy_point_type = scalfmm::container::point<value_type&, dimension>;

    proxy_point_type p4(p1);
    proxy_point_type p5(p2);
    proxy_point_type p6(p3);

    std::cout << "- p4 = " << p4 << std::endl;
    std::cout << "- p5 = " << p5 << std::endl;
    std::cout << "- p6 = " << p6 << std::endl;
    std::cout << "- p4 == p5 ? : " << (p4 == p5) << std::endl;
    std::cout << "- p4 == p6 ? : " << (p4 == p6) << std::endl;
    std::cout << std::endl;

    // Test equality operator with points of SIMD type
    using simd_type = xsimd::simd_type<value_type>;
    using point_simd_type = scalfmm::container::point<simd_type, dimension>;

    simd_type batch1(1);
    simd_type batch2(1);
    simd_type batch3(2);

    point_simd_type p7(batch1);
    point_simd_type p8(batch2);
    point_simd_type p9(batch3);

    std::cout << "- p7 = " << p7 << std::endl;
    std::cout << "- p8 = " << p8 << std::endl;
    std::cout << "- p9 = " << p9 << std::endl;
    std::cout << "- p7 == p8 ? : " << (p7 == p8) << std::endl;
    std::cout << "- p7 == p9 ? : " << (p7 == p9) << std::endl;
    std::cout << std::endl;
}

auto main() -> int
{
    run_equality_point<int, 3>();
    run_equality_point<float, 3>();
    run_equality_point<double, 3>();

    return 0;
}
