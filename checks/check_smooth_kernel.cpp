// scalfmm
#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/debug.hpp"
#include "scalfmm/matrix_kernels/gaussian.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/tree.hpp"
#include "scalfmm/utils/accurater.hpp"

// cpp tools
#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>

// STL
#include <random>
#include <vector>

#define PART_VAR

namespace local_args
{
    struct tree_height
    {
        cpp_tools::cl_parser::str_vec flags = {"--tree-height", "-th"};
        std::string description = "Height of the tree.";
        using type = std::size_t;
        std::string input_hint = "int"; /*!< The input hint */
        type def = 4;
    };

    struct nb_particles
    {
        cpp_tools::cl_parser::str_vec flags = {"--N", "--number-particles"};
        std::string description = "Numbre of particles to generate";
        using type = std::size_t;
        std::string input_hint = "int"; /*!< The input hint */
        type def = 1000;
    };

    struct order
    {
        cpp_tools::cl_parser::str_vec flags = {"--order", "-o"};
        std::string description = "Order of the approximation.";
        using type = std::size_t;
        std::string input_hint = "int"; /*!< The input hint */
        type def = 4;
    };
    struct gauss_coeff
    {
        cpp_tools::cl_parser::str_vec flags = {"--gauss-coeff", "-gc"};
        std::string description = "Coefficient of the gaussian.";
        using type = double;
        std::string input_hint = "double"; /*!< The input hint */
        type def = type(1.0);
    };
    struct gauss_regul
    {
        cpp_tools::cl_parser::str_vec flags = {"--gauss-regul", "-gr"};
        std::string description = "Regularisation Coefficient of the gaussian on the diagonal term.";
        using type = double;
        std::string input_hint = "double"; /*!< The input hint */
        type def = type(0.0);
    };
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
    {
        os << "{ ";
        for(auto el: vec)
        {
            os << el << ' ';
        }
        os << "}";
        return os;
    }

    template<typename ParserType, typename T, typename... Args>
    auto print_parameters(ParserType const& parser, T&& value, Args&&... args) -> void
    {
        T dummy;
        std::cout << cpp_tools::colors::cyan;
        std::cout << std::boolalpha;
        std::cout << "[param] " << std::left << std::setw(24) << dummy.flags[0] << " = " << parser.template get<T>()
                  << "\n";
        if constexpr(sizeof...(args) > 0)
        {
            print_parameters(parser, std::forward<Args>(args)...);
        }
        std::cout << cpp_tools::colors::reset;
    }
}   // namespace local_args

namespace local_utils
{
    // Primary template handles matrix kernels that are not gaussian kernels
    template<typename MatrixKernelType>
    struct is_gaussian_kernel : std::false_type
    {
    };

    // Partial specialization recognizes gaussian kernels
    template<typename MatrixKernelType>
    struct is_gaussian_kernel<scalfmm::matrix_kernels::gaussian<MatrixKernelType>> : std::true_type
    {
    };

    // Helper alias
    template<typename MatrixKernelType>
    inline static constexpr bool is_gaussian_kernel_v = is_gaussian_kernel<MatrixKernelType>::value;
}   // namespace local_utils

template<typename ValueType, std::size_t Dimension, typename MatrixKernelType, typename ParserType>
auto run(ParserType const& parser) -> void
{
    using value_type = ValueType;
    static constexpr std::size_t dimension = Dimension;
    using matrix_kernel_type = MatrixKernelType;

    static constexpr std::size_t nb_inputs{matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs{matrix_kernel_type::kn};

    std::cout << cpp_tools::colors::blue << "dimension: " << dimension << std::endl << cpp_tools::colors::reset;
#ifdef PART_VAR
    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs, std::int64_t>;
#else
    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs>;

#endif
    using position_type = typename particle_type::position_type;
    using box_type = scalfmm::component::box<position_type>;

    using container_type = std::vector<particle_type>;

    // interpolation types
    using near_field_type = scalfmm::operators::near_field_operator<matrix_kernel_type>;
    using interpolator_type = scalfmm::interpolation::interpolator<value_type, dimension, matrix_kernel_type,
                                                                   scalfmm::options::uniform_<scalfmm::options::fft_>>;
    //    scalfmm::options::chebyshev_<scalfmm::options::low_rank_>>;
    using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
    using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    // tree types
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    const std::size_t nb_particles{parser.template get<local_args::nb_particles>()};
    const std::size_t tree_height{parser.template get<local_args::tree_height>()};
    const std::size_t order{parser.template get<local_args::order>()};

    // we construct the tree
    container_type container(nb_particles);

    const value_type box_width{2.};
    const position_type box_center(1.);
    box_type box(box_width, box_center);

    // random generator
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(123);
    std::uniform_real_distribution<value_type> dis(0.0, 2.0);
    auto random_r = [&dis, &gen]() { return dis(gen); };

    // inserting particles in the container
    for(std::size_t idx = 0; idx < nb_particles; ++idx)
    {
        // particle_type p;
        particle_type& p = container[idx];
        for(auto& e: p.position())
        {
            e = random_r();
        }
        for(auto& e: p.inputs())
        {
            e = random_r();
        }
        for(auto& e: p.outputs())
        {
            e = value_type(0.);
        }
#ifdef PART_VAR

        p.variables(idx);
#endif
    }

    // construct the near field
    near_field_type near_field{};

    // a reference on the matrix_kernel of the near_field
    auto& mk = near_field.matrix_kernel();

    // print name of the kernel
    std::cout << "- Kernel name: " << mk.name() << std::endl;
    // if we deal with a gaussian kernel, we set the different parameters
    if constexpr(local_utils::is_gaussian_kernel_v<matrix_kernel_type>)
    {
        value_type coeff{parser.template get<local_args::gauss_coeff>()};
        value_type reg{parser.template get<local_args::gauss_regul>()};
        mk.set_coeff(coeff);
        mk.set_epsilon(reg);
    }

    // test whether the kernel is smooth or not (in that case, just print message to the standard output)
    if constexpr(scalfmm::meta::is_smooth_v<matrix_kernel_type>)
    {
        std::cout << "Kernel is smooth" << std::endl;
    }
    else
    {
        std::cout << "Kernel is not smooth" << std::endl;
    }
    std::cout << " Good order " << int(nb_particles / std::pow(2, tree_height)) << std::endl;
    // build the approximation used in the near field
    interpolator_type interpolator(mk, order, tree_height, box.width(0));
    far_field_type far_field(interpolator);
    //  construct the fmm operator
    fmm_operator_type fmm_operator(near_field, far_field);
    // using settings = typename interpolator_type::settings;
    // settings s;

    std::cout << cpp_tools::colors::blue;
    fmm_operator.settings(std::cout);
    std::cout << cpp_tools::colors::reset;

    const std::size_t group_size{10};   // the number of cells and leaf grouped in the tree
    group_tree_type tree(tree_height, order, box, group_size, group_size, container);

    tree.statistics("smmoth kernel", std::cout);

    scalfmm::list::sequential::build_interaction_lists(tree, tree, fmm_operator);
    // now we have everything to call the fmm algorithm
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](tree, fmm_operator);

    scalfmm::algorithms::full_direct(container, mk);
    //
    scalfmm::utils::accurater<value_type> error{};
#ifdef PART_VAR

    scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                      [&container, &error](auto const& leaf)
                                      {
                                          // loop on the particles of the leaf
                                          for(auto const p_ref: leaf)
                                          {
                                              // build a particle
                                              const auto p = typename leaf_type::const_proxy_type(p_ref);
                                              //
                                              const auto& idx = std::get<0>(p.variables());

                                              auto const& output_ref = container[idx].outputs();
                                              auto const& output = p.outputs();
                                              for(std::size_t i{0}; i < nb_outputs; ++i)
                                              {
                                                  error.add(output_ref.at(i), output.at(i));
                                              }
                                          }
                                      });
    std::cout << cpp_tools::colors::red;
    std::cout << error << '\n';
    std::cout << cpp_tools::colors::reset;

#endif

    std::cout << " write direct computation in data_smmooth_direct.fma\n";
    scalfmm::io::FFmaGenericWriter<value_type> writer("data_smmooth_direct.fma", false);
    writer.writeDataFrom(container, box.center(), tree.box_width());
    std::cout << " write fmm computation in data_smmooth_fmm.fma\n\n";
    scalfmm::io::FFmaGenericWriter<value_type> writer2("data_smmooth_fmm.fma", false);
    writer2.writeDataFromTree(tree, nb_particles);
}

template<typename... Args>
auto get_parser(int argc, char* argv[], Args&&... args)
{
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, std::forward<Args>(args)...);
    parser.parse(argc, argv);
    local_args::print_parameters(parser, std::forward<Args>(args)...);
    std::cout << "\n";

    return parser;
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    using value_type = double;
    static constexpr std::size_t dimension = 2;

    const std::size_t order{8};
    const std::size_t tree_height{4};
    const std::size_t nb_particles{10000};

    // Parameter handling
    auto parser = get_parser(argc, argv, local_args::nb_particles{}, local_args::tree_height{}, local_args::order{},
                             local_args::gauss_coeff(), local_args::gauss_regul());

    // Gaussian kernel
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::gaussian<value_type>;
        run<value_type, dimension, matrix_kernel_type>(parser);
    }
#ifdef ALL_KERNELS
    // One over r (modified)
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::debug::one_over_r_modified;
        run<value_type, dimension, matrix_kernel_type>(parser);
    }
#endif

    // One over r (classical)
    {
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        run<value_type, dimension, matrix_kernel_type>(parser);
    }
    return 0;
}
