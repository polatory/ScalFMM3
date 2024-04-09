// @FUSE_FFTW
#include <fstream>
#include <iostream>

#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/uniform.hpp"
#include "scalfmm/interpolation/chebyshev.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"

using namespace scalfmm;

template<int dimension, typename value_type, typename matrix_kernel_type>
int test_interp(const std::size_t order)
{
    static constexpr std::size_t nb_inputs{matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs{matrix_kernel_type::kn};
    // std::string sinterp("_unif.txt");
    // using interpolator_type = scalfmm::interpolation::uniform_interpolator<value_type, dimension, matrix_kernel_type>;
    std::string sinterp("_cheb.txt");
    using interpolator_type = scalfmm::interpolation::chebyshev_interpolator<value_type, dimension, matrix_kernel_type>;
    
    using point_type = container::point<value_type, dimension>;
    //
    const std::size_t tree_height{3};
    const value_type box_width{1.};

    interpolator_type interpolator(matrix_kernel_type{}, order, tree_height, box_width);

    auto roots = interpolator.roots();
    std::cout << "roots " << roots << std::endl;
    const int N = 80;
    auto points = xt::linspace(value_type(-1.), value_type(1), N);

    std::vector<value_type> poly_of_part(N);
    std::vector<value_type> der_poly_of_part(N);
    // generate function atp = 3
    std::size_t p = order/2;
    // generate the pth lagrange polynomial and its derivative

    for(std::size_t part = 0; part < points.size(); ++part)
    {

        poly_of_part[part] = interpolator.polynomials(points[part], p);

        der_poly_of_part[part] = interpolator.derivative(points[part], p);
        std::clog << part << " " << points[part] << "  " << poly_of_part[part]
                  << std::endl;
    }

    //
    // Save the roots in a file
    std::ofstream out("roots"+sinterp);

    for(std::size_t part = 0; part < roots.size(); ++part)
    {
        // generate polynomials

        out << roots[part] << std::endl;
    }
    out.close();
    // save basis function p and its derivative in a file

    std::ofstream out_p("points"+sinterp);

    for(std::size_t part = 0; part < points.size(); ++part)
    {
        out_p << points[part] << " " << poly_of_part[part] << " " << der_poly_of_part[part] << std::endl;
    }
    out_p.close();

    std::ofstream out_p1("lagrange"+sinterp);

    for (std::size_t part = 0; part < points.size(); ++part) {
      out_p1 << points[part] << " ";
      for (int n = 0; n < order; ++n) {
        out_p1 << interpolator.polynomials(points[part], n) << " ";
      }
      out_p1  << std::endl;
    }
    out_p1.close();

    return 0;
}

int main(int argc, char* argv[])
{
    using value_type = double;
    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

    return test_interp<1, value_type, matrix_kernel_type>(7);
}
