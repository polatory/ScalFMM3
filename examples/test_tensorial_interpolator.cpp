
#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>

#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <scalfmm/container/point.hpp>
#include <scalfmm/container/variadic_adaptor.hpp>
#include <scalfmm/kernels/generic/operators.hpp>
#include <scalfmm/kernels/generic/p2p.hpp>
#include <scalfmm/tags/tags.hpp>
//#include <scalfmm/interpolation/chebyshev.hpp>
#include <scalfmm/interpolation/uniform.hpp>
#include <scalfmm/matrix_kernels/kernels.hpp>
#include <scalfmm/meta/utils.hpp>
#include <scalfmm/tools/colorized.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <scalfmm/utils/math.hpp>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>

auto main() -> int
{
    using size_type = std::size_t;
    namespace colors = scalfmm::colors;

    constexpr size_type particle_dim{3};
    constexpr size_type physical_values_dim{1};
    constexpr size_type order{3};
    constexpr size_type nnodes{scalfmm::math::pow(order, particle_dim)};

    using value_type = double;
    using point_type = scalfmm::container::point<double, particle_dim>;
    using particle_type = scalfmm::container::particle<double, particle_dim, double>;   //, double, double>;
    using output_type = std::tuple<double>;                                             //, double, double>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type, output_type, scalfmm::operators::tags::with_forces>;
    using cell_type = scalfmm::component::cell<double, particle_dim, physical_values_dim, std::complex<double>>;

    // using interpolator_type = scalfmm::interpolation::chebyshev_interpolator<double, particle_dim>;
    using matrix_kernel_type = scalfmm::matrix_kernels::one_over_r;
    using interpolator_type =
      scalfmm::interpolation::uniform_interpolator<matrix_kernel_type, value_type, particle_dim>;

    const size_type nb_particles{4};
    const size_type tree_height{5};
    const double width{1.0};

    container_type particles_source(nb_particles);
    container_type particles_source_other(nb_particles);

    std::mt19937 gen(33);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    auto random_r = [&dis, &gen]() { return dis(gen); };

    point_type center = {2. * width, 0., 0.};

    auto make_particle = [&center, &width, &random_r]() {
        point_type position = {random_r() * width, random_r() * width, random_r() * width};
        position += center;
        particle_type p(position, random_r());   //, random_r(), random_r());
        return p.as_tuple();
    };

    std::generate(particles_source.begin(), particles_source.end(), make_particle);
    std::generate(particles_source_other.begin(), particles_source_other.end(), make_particle);

    auto start = std::chrono::high_resolution_clock::now();
    interpolator_type s(order, tree_height);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << colors::green << "Interpolator construction time : "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us\n"
              << colors::reset;

    // Construct leaf
    leaf_type l(std::move(particles_source), nb_particles, center, width, 0);
    leaf_type l_other(std::move(particles_source_other), nb_particles, center, width, 0);

    // Construct cells
    cell_type child_cell(scalfmm::container::point<std::int64_t, 3>(0), width, order);
    cell_type target_cell(scalfmm::container::point<std::int64_t, 3>(0), width, order);
    // scalfmm::meta::for_each(child_cell.multipoles(), [](auto& t) { t.resize({order, order, order}); });
    // scalfmm::meta::for_each(child_cell.multipoles(), [](auto& t) { std::fill(t.begin(), t.end(), value_type(0.)); });

    cell_type parent_cell(scalfmm::container::point<std::int64_t, 3>(0), width, order);
    // scalfmm::meta::for_each(parent_cell.multipoles(), [](auto& t) { t.resize({order, order, order}); });
    // scalfmm::meta::for_each(parent_cell.multipoles(), [](auto& t) { std::fill(t.begin(), t.end(), value_type(0.));
    // });

    // Lambda for output result to file
    auto print_to_file = [](auto& file, auto& container, std::size_t size) {
        auto it = std::begin(container);
        for(std::size_t i = 0; i < size; ++i)
        {
            file << std::fixed << std::setprecision(13) << *it << '\n';
            it++;
        }
    };

    // P2M test
    std::vector<std::chrono::microseconds> t;
    for(std::size_t exp = 0; exp < 1; ++exp)
    {
        start = std::chrono::high_resolution_clock::now();
        scalfmm::operators::apply_p2m(s, l, child_cell.multipoles(), order);
        end = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));

        if(exp == 0)
        {
            std::ofstream file("dump_p2m.txt");
            scalfmm::meta::for_each(child_cell.multipoles(),
                                    [&file, &nnodes, &print_to_file](auto& t) { print_to_file(file, t, nnodes); });
            file.close();
        }
    }

    std::sort(t.begin(), t.end());

    std::cout << colors::yellow << "p2m took... : " << t[t.size() / 2].count() << "us\n" << colors::reset;

    //    // M2M test
    //
    //    t.clear();
    //    for(std::size_t exp = 0; exp < 1; ++exp)
    //    {
    //        start = std::chrono::high_resolution_clock::now();
    //        scalfmm::operators::apply_m2m(s, child_cell.multipoles(), 0, parent_cell.multipoles(), order, 2);
    //        end = std::chrono::high_resolution_clock::now();
    //        t.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    //
    //        if(exp == 0)
    //        {
    //            std::ofstream file_m2m("dump_m2m.txt");
    //            scalfmm::meta::for_each(child_cell.multipoles(), [&file_m2m, &nnodes, &print_to_file](auto& t) {
    //                print_to_file(file_m2m, t, nnodes);
    //            });
    //            file_m2m.close();
    //        }
    //    }
    //
    //    std::sort(t.begin(), t.end());
    //
    //    std::cout << colors::cyan << "m2m took... : " << t[t.size() / 2].count() << "us\n" << colors::reset;
    //
    //    // L2L test
    //
    //    t.clear();
    //
    //    // scalfmm::meta::for_each(parent_expansion, [](auto& t){ std::fill(t.begin(), t.end(), value_type(0.)); } );
    //
    //    for(std::size_t exp = 0; exp < 1; ++exp)
    //    {
    //        start = std::chrono::high_resolution_clock::now();
    //        scalfmm::operators::apply_l2l(s, child_cell.multipoles(), 0, parent_cell.multipoles(), order, 2);
    //        end = std::chrono::high_resolution_clock::now();
    //        t.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    //
    //        if(exp == 0)
    //        {
    //            std::ofstream file_l2l("dump_l2l.txt");
    //            scalfmm::meta::for_each(child_cell.multipoles(), [&file_l2l, &nnodes, &print_to_file](auto& t) {
    //                print_to_file(file_l2l, t, nnodes);
    //            });
    //            file_l2l.close();
    //        }
    //    }
    //
    //    std::sort(t.begin(), t.end());
    //
    //    std::cout << colors::magenta << "l2l took... : " << t[t.size() / 2].count() << "us\n" << colors::reset;

    // M2L test
    scalfmm::operators::m2m(s, child_cell, 0, parent_cell, order, 2);

    // memory check !
    scalfmm::operators::m2l(s, parent_cell, 9, target_cell, order, 2);

    std::vector<std::unique_ptr<leaf_type>> neighbor;
    neighbor.push_back(std::make_unique<leaf_type>(std::move(particles_source_other), nb_particles, center, width, 0));

    start = std::chrono::high_resolution_clock::now();
    scalfmm::operators::p2p_full_mutual(s, l, neighbor, scalfmm::operators::tags::with_forces{});
    end = std::chrono::high_resolution_clock::now();

    std::cout << colors::italic << colors::yellow
              << "p2p took... : " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << "us\n"
              << colors::reset;

    // std::ofstream file("dump_forces.txt");
    // scalfmm::meta::for_each(l.cforces(),
    //        [&file, &nnodes, &print_to_file](auto& t) { print_to_file(file, t, nnodes); });
    // file.close();

    // std::ofstream file_("dump_positions.txt");
    // scalfmm::meta::for_each(l.cparticles(),
    //        [&file_, &nnodes, &print_to_file](auto& t) { print_to_file(file_, t, nnodes); });
    // file_.close();

    return 0;
}
