// @FUSE_FFTW
// @FUSE_CBLAS

///
/// For uniform
/// ./check/Release/check_l2p  --input-file ../data/datal2p_10.fma --order 4
///
/// For cheb
/// ./check/Release/check_l2p  --input-file ../data/datal2p_10.fma --order 4 -cheb

#include <iostream>
#include <string>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/operators/interpolation/l2p.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/parameters.hpp"
#include <cpp_tools/colors/colorized.hpp>

using value_type = double;

namespace functions
{
    constexpr value_type fourpi = 12.56637;
    auto cos = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return std::cos(fourpi * (x * y + std::pow(z, 4)));
    };
    auto derx_cos = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return -fourpi * y * std::sin(fourpi * (x * y + std::pow(z, 4)));
    };
    auto dery_cos = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return -fourpi * x * std::sin(fourpi * (x * y + std::pow(z, 4)));
    };
    auto derz_cos = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return -fourpi * value_type(4.0) * std::pow(z, 3) * std::sin(fourpi * (x * y + std::pow(z, 4)));
    };

    constexpr value_type alpha = 4.0;

    auto exp = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return std::exp(-alpha * (x + y + z));
    };
    auto derx_exp = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return -alpha * std::exp(-alpha * (x + y + z));
    };
    auto dery_exp = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return -alpha * std::exp(-alpha * (x + y + z));
    };
    auto derz_exp = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return -alpha * std::exp(-alpha * (x + y + z));
    };

    auto poly = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return x * x + x * y + (y + 1) * (1 + y) + std::pow(z + 1, 4);
    };
    auto derx_poly = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return value_type(2.0) * x + y;
    };
    auto dery_poly = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return x + 2 * (y + 1);
    };
    auto derz_poly = [](const value_type& x, const value_type& y, const value_type& z) -> value_type {
        return value_type(4.0) * std::pow(z + 1, 3);
    };
}   // namespace functions
template<typename far_field_type>
auto run(const std::string& title, const std::string& input_file, const int& tree_height, const int& group_size,
         const int& order) -> int
{
    std::cout << "run: " << title << std::endl;
    static constexpr std::size_t dimension = 3;
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    // Open particle file
    std::size_t number_of_particles{};

    // ---------------------------------------
    // scalfmm 3.0 tree tests and benchmarks.
    // ---------------------------------------
    using matrix_kernel_type = typename far_field_type::approximation_type::matrix_kernel_type;
    using interpolator_type = typename far_field_type::approximation_type;
    static constexpr std::size_t nb_inputs{matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs{matrix_kernel_type::kn};
    // Number of output p + force
    static constexpr std::size_t nb_outputs_leaf{nb_outputs + dimension};

    // pos, charge, pot + forces no variables
    using particle_type = scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type,
                                                       nb_outputs_leaf, std::size_t>;
    using container_type = scalfmm::container::particle_container<particle_type>;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using position_type = typename particle_type::position_type;
    //
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    std::cout << cpp_tools::colors::green << "Creating & Inserting " << number_of_particles
              << "particles for version 3.0 ...\n"
              << cpp_tools::colors::reset;

    scalfmm::container::point<double, 3> box3_center{};

    double box_width{};
    //{
    bool verbose = true;

    scalfmm::io::FFmaGenericLoader<double> loader(input_file, verbose);

    box3_center = scalfmm::container::point<double, 3>{loader.getBoxCenter()[0], loader.getBoxCenter()[1],
                                                       loader.getBoxCenter()[2]};
    number_of_particles = loader.getNumberOfParticles();
    box_width = loader.getBoxWidth();
    std::cout << cpp_tools::colors::green << "Box center = " << box3_center << cpp_tools::colors::reset << '\n';
    //	point3_type cornerMin{100.0, 100.0,100.0},cornerMax{};
    container_type container(number_of_particles);

    auto nb_val_to_red_per_part = loader.get_dimension() + loader.get_number_of_input_per_record();
    double* values_to_read = new double[nb_val_to_red_per_part]{};
    for(std::size_t idx = 0; idx < number_of_particles; ++idx)
    {
        loader.fillParticle(values_to_read, nb_val_to_red_per_part);
        particle_type p;
        std::size_t ii{0};
        for(auto& e: p.position())
        {
            e = values_to_read[ii++];
        }
        for(auto& e: p.inputs())
        {
            e = 0.0;
            // values_to_read[ii++];
        }
        for(auto& e: p.outputs())
        {
            e = 0.;
        }
        // std::clog << "idx: " << idx << "  part: " << p << std::endl;
        container.insert_particle(idx, p);
    }
    loader.close();
    //    }

    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::yellow << "Container loaded \n" << cpp_tools::colors::reset;

    box_type box(box_width, box3_center);
    group_tree_type gtree(static_cast<std::size_t>(tree_height), order, box, static_cast<std::size_t>(group_size),
                          static_cast<std::size_t>(group_size), container);
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //// construct the interpolator and the far-field
    ///
    interpolator_type interpolator(matrix_kernel_type{}, order, static_cast<std::size_t>(tree_height), box.width(0));
    far_field_type far_field(interpolator);

    ///
    /////////////////////////////////////////////////////////////////////////////

    {
        std::cout << std::endl << " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& " << std::endl;
        std::cout << std::scientific;
        std::cout.precision(15);

        //
        auto roots_1d = interpolator.roots();
        std::cout << "roots: " << roots_1d << std::endl;

        auto roots = scalfmm::tensor::generate_meshgrid<dimension>(roots_1d);
        const auto roots_x = xt::flatten(std::get<0>(roots));
        const auto roots_y = xt::flatten(std::get<1>(roots));
        const auto roots_z = xt::flatten(std::get<2>(roots));

        int pos = 0, global_idx = 0;

        scalfmm::component::for_each_cell_leaf(
          std::begin(gtree), std::end(gtree), static_cast<std::size_t>(tree_height),
          [&pos, &order, &roots_x, &roots_y, &roots_z, &global_idx, &far_field](auto& cell, auto& leaf) {
              auto func = functions::poly;
              auto derx_func = functions::derx_poly;
              auto dery_func = functions::dery_poly;
              auto derz_func = functions::derz_poly;

              //   auto source_particle_iterator = leaf.particles().begin();
              //   auto cont = leaf.particles();
              std::cout << "\n  Leaf morton index " << leaf.index() << "  num  " << pos << std::endl;
              std::cout << "\n       center  " << leaf.center() << "  width  " << leaf.width() << std::endl;
              std::cout << "\n  Cell morton index " << cell.index() << "  num  " << pos << std::endl;
              std::cout << "\n       center  " << cell.center() << "  width  " << cell.width() << std::endl;
              // Move roots inside cells   x  = center + width/2 roots
              auto center = cell.center();
              auto half_width = cell.width() * 0.5;
              // Set the collocation points inside the cell
              auto pos_x = half_width * roots_x + center[0];
              auto pos_y = half_width * roots_y + center[1];
              auto pos_z = half_width * roots_z + center[2];

              //   Set the local expansion by  f(pos_x, pos_y, pos_z)

              auto& local_expansion = cell.locals().at(0);
              auto flatten_view = xt::flatten(local_expansion);
              for(int i = 0; i < local_expansion.size(); ++i)
              {
                  flatten_view.at(i) = func(pos_x[i], pos_y[i], pos_z[i]);
              }

              //
              // Apply the l2p operator
              /// far-field operator
              ///
              scalfmm::operators::apply_l2p(far_field, cell, leaf);
#ifndef toto
              //  for the next call  nb_outputs_leaf =  nb_outputs + dimension;
              // scalfmm::operators::apply_l2p(far_field, cell.clocals(), leaf, order);
              //
              // Check the value on the particles

              int idx{0};
              for(auto const p_tuple_ref: leaf)
              {
                  //   scalfmm::meta::td<decltype(p_tuple_ref)> t;
                  // We construct a particle type for classical acces
                  auto proxy = typename group_tree_type::leaf_type::const_proxy_type(p_tuple_ref);
                  std::cout << idx << " p " << proxy << std::endl;
                  // apply func on the position
                  value_type result_of_func[nb_outputs_leaf];
                  auto const& position = proxy.position();
                  result_of_func[0] = std::apply(func, scalfmm::meta::to_tuple(position));
                  result_of_func[1] = std::apply(derx_func, scalfmm::meta::to_tuple(position));
                  result_of_func[2] = std::apply(dery_func, scalfmm::meta::to_tuple(position));
                  result_of_func[3] = std::apply(derz_func, scalfmm::meta::to_tuple(position));

                  auto const& out = proxy.outputs();
                  std::cout << pos << "  " << idx << position << "\n    f(x)=   " << result_of_func[0] << "  " << out[0]
                            << "  diff= " << std::abs(result_of_func[0] - out[0]) / result_of_func[0]
                            << "\n    f_x(x)= " << result_of_func[1] << "  " << out[1]
                            << "  diff= " << std::abs(result_of_func[1] - out[1]) / result_of_func[1]
                            << "\n    f_y(x)= " << result_of_func[2] << "  " << out[2]
                            << "  diff= " << std::abs(result_of_func[2] - out[2]) / result_of_func[2]
                            << "\n    f_z(x)= " << result_of_func[3] << "  " << out[3]
                            << "  diff= " << std::abs(result_of_func[3] - out[3]) / result_of_func[3] << std::endl;
                  ++global_idx;
              }
#endif

              ++pos;
          });
    }
    return 0;
}
struct chebInterp
{
    /// Unused type, mandatory per interface specification
    using type = bool;
    /// The parameter is a flag, it doesn't expect a following value
    enum
    {
        flagged
    };
    cpp_tools::cl_parser::str_vec flags = {
      "--cheb",
    };
    std::string description = "Use Chebychev interpolator raher than uniform one";
};

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(args::tree_height{}, args::order{}, args::input_file{}, chebInterp{},
                                           cpp_tools::cl_parser::help{});

    parser.parse(argc, argv);

    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    const int tree_height{parser.get<args::tree_height>()};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset << '\n';

    const int group_size{1};
    std::cout << cpp_tools::colors::blue << "<params> Group Size : " << group_size << cpp_tools::colors::reset << '\n';

    const std::string input_file{parser.get<args::input_file>()};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset << '\n';
    }
    const bool useCheb(parser.exists<chebInterp>());

    static constexpr std::size_t dimension = 3;
    const auto order{parser.get<args::order>()};
    std::cout << cpp_tools::colors::blue << "<params> Runtime order : " << order << cpp_tools::colors::reset << '\n';

    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;

    std::string title;
    int val{};
    if(useCheb)
    {
        title = " Chebychev interpolator ";
        using interpolator_type =
          scalfmm::interpolation::interpolator<value_type, dimension, matrix_kernel_type,
                                               scalfmm::options::chebyshev_<scalfmm::options::low_rank_>>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;
        val = run<far_field_type>(title, input_file, tree_height, group_size, order);
    }
    else
    {
        title = " Equispaced interpolator ";

        using interpolator_type =
          scalfmm::interpolation::interpolator<value_type, dimension, matrix_kernel_type,
                                               scalfmm::options::uniform_<scalfmm::options::fft_>>;
        using far_field_type = scalfmm::operators::far_field_operator<interpolator_type, true>;

        val = run<far_field_type>(title, input_file, tree_height, group_size, order);
    }

    return val;
}
