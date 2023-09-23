#include <cmath>
#include <iomanip>
#include <limits>

#include "scalfmm/interpolation/builders.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/debug.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/operators/m2l.hpp"
#include "scalfmm/tree/cell.hpp"

#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>

struct find_all_perms
{
    cpp_tools::cl_parser::str_vec flags = {"--find-all"};
    std::string description = "Find all permutations.";
    using type = std::size_t;
    type def = 0;
};

struct source_index_1 : cpp_tools::cl_parser::required_tag
{
    cpp_tools::cl_parser::str_vec flags = {"--source1-index", "-s1"};
    std::string description = "The linear interaction index of source 1.";
    using type = std::size_t;
    type def = 0;
};

struct source_index_2 : cpp_tools::cl_parser::required_tag
{
    cpp_tools::cl_parser::str_vec flags = {"--source2-index", "-s2"};
    std::string description = "The linear interaction index of source 2.";
    using type = std::size_t;
    type def = 0;
};

template<std::size_t kn, std::size_t km, typename Cell, typename ArrayScaleFactor, typename InteractionMatrice,
         typename Permutation>
inline auto apply_m2l_debug(Cell const& source_cell, Cell& target_cell, InteractionMatrice const& interaction_matrices,
                            Permutation const& perm, ArrayScaleFactor scale_factor, std::size_t tree_level,
                            std::size_t nnodes) -> void
{
    // get the transformed multipoles
    auto const& multipoles = source_cell.cmultipoles();
    auto& locals = target_cell.locals();

    auto permuted_multipoles = std::decay_t<decltype(multipoles.at(0))>::from_shape(multipoles.at(0).shape());
    auto permuted_locals = std::decay_t<decltype(locals.at(0))>::from_shape(locals.at(0).shape());

    for(std::size_t n = 0; n < kn; ++n)
    {
        for(std::size_t m = 0; m < km; ++m)
        {
            for(std::size_t i{0}; i < nnodes; ++i)
            {
                permuted_multipoles.data()[perm.at(i)] = multipoles.at(m).data()[i];
            }
            // knm is the interaction between the m_iest source cell and the n_iest target cell.
            auto const& knm = interaction_matrices.at(n, m);
            // locals.at(n) = scale_factor*K_nm*multipoles.at(m) + 1.0*locals.at(n)
            cxxblas::gemv<xt::blas_index_t>(xt::get_blas_storage_order(locals.at(n)),   // locals.at(n)
                                            cxxblas::Transpose::NoTrans, static_cast<xt::blas_index_t>(knm.shape()[0]),
                                            static_cast<xt::blas_index_t>(knm.shape()[1]), scale_factor.at(n),
                                            knm.data() + knm.data_offset(), xt::get_leading_stride(knm),
                                            permuted_multipoles.data() +
                                              permuted_multipoles.data_offset(),   // multipoles.at(m)
                                            1, double(0.), permuted_locals.data() + permuted_locals.data_offset(), 1);

            for(std::size_t i{0}; i < nnodes; ++i)
            {
                locals.at(n).data()[i] += permuted_locals.data()[perm.at(i)];
            }
        }
    }
}

template<typename V, std::size_t D, typename MK>
using interpolator_alias =
  scalfmm::interpolation::interpolator<V, D, MK, scalfmm::options::chebyshev_<scalfmm::options::dense_>>;

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    auto parser = cpp_tools::cl_parser::make_parser(source_index_1{}, source_index_2{}, find_all_perms{},
                                                    cpp_tools::cl_parser::help{});
    parser.parse(argc, argv);

    static constexpr std::size_t dimension{1};
    const std::size_t order{7};
    using value_type = double;
    using matrix_kernel_nsym_type = scalfmm::matrix_kernels::debug::one_over_r_non_symmetric;
    using matrix_kernel_sym_type = scalfmm::matrix_kernels::laplace::one_over_r;
    static constexpr auto km{matrix_kernel_nsym_type::km};
    static constexpr auto kn{matrix_kernel_nsym_type::kn};
    using point_type = scalfmm::container::point<value_type, dimension>;
    using storage_type = scalfmm::component::grid_storage<value_type, dimension, kn, km>;
    using interpolator_nsym_type = interpolator_alias<value_type, dimension, matrix_kernel_nsym_type>;
    using interpolator_sym_type = interpolator_alias<value_type, dimension, matrix_kernel_sym_type>;
    using buffer_type = typename scalfmm::interpolation::interpolator_traits<interpolator_sym_type>::buffer_type;
    // interpolator
    interpolator_nsym_type interpolator_nsym(matrix_kernel_nsym_type{}, order, std::size_t(3), value_type(14.));
    interpolator_sym_type interpolator_sym(matrix_kernel_sym_type{}, order, std::size_t(3), value_type(14.));

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);

    // construct target cell
    scalfmm::component::cell<storage_type> target_ref1(point_type(value_type(0.)), value_type(2.), order);
    scalfmm::component::cell<storage_type> target_sym1(point_type(value_type(0.)), value_type(2.), order);
    scalfmm::component::cell<storage_type> target_ref2(point_type(value_type(0.)), value_type(2.), order);
    scalfmm::component::cell<storage_type> target_sym2(point_type(value_type(0.)), value_type(2.), order);
    // construct 2 source cells
    scalfmm::component::cell<storage_type> source1(point_type(value_type(0.)), value_type(2.), order);
    scalfmm::component::cell<storage_type> source2(point_type(value_type(0.)), value_type(2.), order);

    auto e_sym = interpolator_sym.buffer_initialization();
    auto e_nsym = interpolator_nsym.buffer_initialization();

    if(parser.get<find_all_perms>())
    {
        for(std::size_t m{0}; m < km; ++m)
        {
            std::cout << cpp_tools::colors::blue << "Source 1 multipoles : ";
            source1.multipoles().at(m) = xt::arange<double>(0, scalfmm::math::pow(order, dimension), 1)
                                           .reshape(source1.cmultipoles().at(m).shape());
            std::cout << source1.cmultipoles().at(m) << '\n';
            std::cout << cpp_tools::colors::yellow << "Source 2 multipoles : ";
            source2.multipoles().at(m) = xt::arange<double>(0, scalfmm::math::pow(order, dimension), 1)
                                           .reshape(source2.cmultipoles().at(m).shape());
            std::cout << source2.cmultipoles().at(m) << '\n';
        }

        auto perms_and_k_indices = scalfmm::interpolation::get_permutations_and_indices<dimension>(
          order, interpolator_sym.nnodes(), interpolator_sym.m2l_interactions());
        auto const& permutations = std::get<0>(perms_and_k_indices);
        // auto const& k_indices = std::get<1>(perms_and_k_indices);

        std::size_t flat_index{0};
        bool allok{true};
        auto generate_all_perms = [&](auto... is)
        {

                for(std::size_t n{0}; n < kn; ++n)
                {
                    target_sym1.locals().at(n) = xt::zeros_like(target_sym1.clocals().at(n));
                }
                scalfmm::operators::m2l(interpolator_nsym, source1, flat_index, target_ref1, std::size_t(3), e_nsym);

                auto k = interpolator_sym.interactions_matrices().at(interpolator_sym.symmetry_k_index(flat_index));

                auto p = xt::eval(xt::view(permutations, flat_index, xt::all()));

                apply_m2l_debug<kn, km>(source1, target_sym1, k, p,
                                        matrix_kernel_sym_type{}.scale_factor(target_sym1.width()), std::size_t(3),
                                        interpolator_sym.nnodes());

                bool ok{true};
                for(std::size_t n{0}; n < kn; ++n)
                {
                    if(xt::allclose(target_ref1.clocals().at(n), target_sym1.clocals().at(n)))
                    {
                        ok &= true;
                        allok &= true;
                    }
                    else
                    {
                        ok &= false;
                    }
                }
                if(ok)
                {
                    std::cout << cpp_tools::colors::blue;
                    std::cout << "Symmetries for flat index " << flat_index << " ok.\n";
                }
                else
                {
                    std::size_t cmp_k{0};
                    std::size_t cmp_p{0};
                    bool found{true};

                    for(std::size_t ik{0}; ik < scalfmm::interpolation::number_of_matrices_in_orthant<dimension>();
                        ++ik)
                    {
                        for(std::size_t i{0}; i < interpolator_sym.m2l_interactions(); ++i)
                        {
                            for(std::size_t n{0}; n < kn; ++n)
                            {
                                target_sym1.locals().at(n) = xt::zeros_like(target_sym1.clocals().at(n));
                            }
                            auto k_search = interpolator_sym.interactions_matrices().at(ik);
                            auto p_search = xt::eval(xt::view(permutations, i, xt::all()));
                            apply_m2l_debug<kn, km>(source1, target_sym1, k_search, p_search,
                                                    matrix_kernel_sym_type{}.scale_factor(target_sym1.width()),
                                                    std::size_t(3), interpolator_sym.nnodes());
                            for(std::size_t n{0}; n < kn; ++n)
                            {
                                if(xt::allclose(target_ref1.clocals().at(n), target_sym1.clocals().at(n)))
                                {
                                    found &= true;
                                }
                                else
                                {
                                    found &= false;
                                }
                            }
                            ++cmp_p;
                        }
                        ++cmp_k;
                    }
                    if(found)
                    {
                        std::cout << cpp_tools::colors::cyan;
                        std::cout << "Symmetries for flat index " << flat_index << "should be : " << cmp_p - 1 << ".\n";
                        std::cout << "k for flat index " << flat_index << "should be : " << cmp_k - 1 << ".\n";
                    }
                    else
                    {
                        std::cout << cpp_tools::colors::red;
                        std::cout << "Can't find symmetry for flat index " << flat_index << ".\n";
                    }
                }

            for(std::size_t n{0}; n < kn; ++n)
            {
                target_ref1.locals().at(n) = xt::zeros_like(target_ref1.clocals().at(n));
            }
            ++flat_index;
        };

        // loop range [-3,4[, ie range concept exclude the last value.
        std::array<int, dimension> starts{};
        std::array<int, dimension> stops{};
        starts.fill(-3);
        stops.fill(4);
        // he we expand at compile time d loops of the range
        // the indices of the d loops are input parameters of the lambda generate_all_interactions
        scalfmm::meta::looper_range<dimension>{}(generate_all_perms, starts, stops);
    }
    else
    {
        for(std::size_t m{0}; m < km; ++m)
        {
            std::cout << cpp_tools::colors::blue << "Source 1 multipoles : ";
            source1.multipoles().at(m) = xt::arange<double>(0, scalfmm::math::pow(order, dimension), 1)
                                           .reshape(source1.cmultipoles().at(m).shape());
            std::cout << source1.cmultipoles().at(m) << '\n';
            std::cout << cpp_tools::colors::yellow << "Source 2 multipoles : ";
            source2.multipoles().at(m) = xt::arange<double>(0, scalfmm::math::pow(order, dimension), 1)
                                           .reshape(source2.cmultipoles().at(m).shape());
            std::cout << source2.cmultipoles().at(m) << '\n';
        }

        // (2,0,0)
        std::size_t source1_index{parser.get<source_index_1>()};
        // (0,2,0)
        std::size_t source2_index{parser.get<source_index_2>()};
        // calling m2l
        std::cout << cpp_tools::colors::green;

        scalfmm::operators::m2l(interpolator_nsym, source1, source1_index, target_ref1, std::size_t(3), e_nsym);

        for(std::size_t n{0}; n < kn; ++n)
        {
            std::cout << cpp_tools::colors::magenta << "Reference locals for source 1 :\n";
            std::cout << xt::flatten(target_ref1.clocals().at(n)) << '\n';
        }

        std::cout << cpp_tools::colors::green;
        scalfmm::operators::m2l(interpolator_nsym, source2, source2_index, target_ref2, std::size_t(3), e_nsym);

        for(std::size_t n{0}; n < kn; ++n)
        {
            std::cout << cpp_tools::colors::magenta << "Reference locals for source 2 :\n";
            std::cout << xt::flatten(target_ref2.clocals().at(n)) << '\n';
        }

        // call_symmetries
        std::cout << cpp_tools::colors::green;
        scalfmm::operators::m2l(interpolator_sym, source1, source1_index, target_sym1, std::size_t(3), e_sym);

        for(std::size_t n{0}; n < kn; ++n)
        {
            std::cout << cpp_tools::colors::magenta << "Symmetries locals for source 1 :\n";
            std::cout << xt::flatten(target_sym1.clocals().at(n)) << '\n';
        }

        std::cout << cpp_tools::colors::green;
        scalfmm::operators::m2l(interpolator_sym, source2, source2_index, target_sym2, std::size_t(3), e_sym);

        for(std::size_t n{0}; n < kn; ++n)
        {
            std::cout << cpp_tools::colors::magenta << "Symmetries locals for source 2 :\n";
            std::cout << xt::flatten(target_sym2.clocals().at(n)) << '\n';
        }

        for(std::size_t n{0}; n < kn; ++n)
        {
            if(xt::allclose(target_ref1.clocals().at(n), target_sym1.clocals().at(n)))
            {
                std::cout << cpp_tools::colors::blue;
                std::cout << "Symmetries for source1 are ok.\n";
            }
            else
            {
                std::cout << cpp_tools::colors::red;
                std::cout << "Symmetries for source1 are wrong !.\n";
            }

            if(xt::allclose(target_ref2.clocals().at(n), target_sym2.clocals().at(n)))
            {
                std::cout << cpp_tools::colors::blue;
                std::cout << "Symmetries for source2 are ok.\n";
            }
            else
            {
                std::cout << cpp_tools::colors::red;
                std::cout << "Symmetries for source2 are wrong !.\n";
            }
        }
    }

    std::cout << cpp_tools::colors::reset;

    return 0;
}
