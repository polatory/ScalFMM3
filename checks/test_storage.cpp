#include <scalfmm/memory/storage.hpp>
#include <iostream>
#include <xtensor/xio.hpp>
#include <scalfmm/matrix_kernels/laplace.hpp>
#include <scalfmm/interpolation/interpolator.hpp>
#include <scalfmm/interpolation/uniform/uniform_interpolator.hpp>

using namespace scalfmm;


auto main() -> int
{
    static constexpr std::size_t dimension{2};
    static constexpr std::size_t kn{2};
    static constexpr std::size_t km{2};

    //memory::tensor_storage<double, dimension, kn, km> t({2,2});
    //std::cout << t.get(0) << '\n';
    //component::uniform_fft_storage<double, 2, km, kn> m(std::size_t(8));
    //component::uniform_fft_storage<double, 2, km, kn> m_m = std::move(m);
    //auto& mref = m.transformed_multipoles();
    //auto const& cmref = m.transformed_multipoles();
    //auto& mrefi1 = m.transformed_multipoles(1);
    //auto const& cmrefi0 = m.transformed_multipoles(0);
    memory::aggregate_storage<memory::multipoles_storage<double, dimension, km>//,
                              //memory::locals_storage<double, dimension, kn>//,
                              //memory::transformed_multipoles_storage<double, dimension, km>
                              >
      a(std::size_t(8));
    memory::aggregate_storage<memory::multipoles_storage<double, dimension, km>//,
                              //memory::locals_storage<double, dimension, kn>//,
                              //memory::transformed_multipoles_storage<double, dimension, km>
                              >
                              a_m = std::move(a);

    //std::cout << memory::check_dimensions<2,2,2>() << '\n';

    //std::cout << a.multipoles() << '\n';
    //std::cout << a.locals() << '\n';
    //std::cout << a.transformed_multipoles() << '\n';

    //interpolation::uniform_fft_interpolator<double, dimension, matrix_kernels::laplace::grad_one_over_r<dimension>> u_(
    //  matrix_kernels::laplace::grad_one_over_r<dimension>{}, 4, 4);

    return 0;
}
