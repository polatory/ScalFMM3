// --------------------------------
// See LICENCE file at project root
// File : algorithm/fmm.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_FMM_HPP
#define SCALFMM_ALGORITHMS_FMM_HPP
#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/sequential/sequential.hpp"
#include "scalfmm/options/options.hpp"
#ifdef _OPENMP
#include "scalfmm/algorithms/omp/task_dep.hpp"
#endif

namespace scalfmm::algorithms
{
    namespace impl
    {
        template<typename Tree, typename NearField, typename FarField>
        auto fmm(Tree& tree, operators::fmm_operators<NearField, FarField> const& fmmoperators, unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential::sequential(tree, fmmoperators, op);
        }
        template<typename TreeS, typename TreeT, typename FmmOperators>
        auto fmm(TreeS& tree_source, TreeT& tree_target, FmmOperators const& fmmoperators,
                 unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential::sequential(tree_source, tree_target, fmmoperators, op);
        }
        template<typename... S, typename Tree, typename FmmOperators>
        inline auto fmm(options::settings<S...> s, Tree& tree, FmmOperators const& fmmoperators,
                             unsigned int op = operators_to_proceed::all) -> void
        {
            static_assert( options::support(s, options::_s(options::omp, options::omp_timit, options::seq, options::seq_timit))
                         , "unsupported fmm algo options!");
            if constexpr (options::has(s, options::seq, options::seq_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return sequential::sequential[inner_settings{}](tree, fmmoperators, op);
            }
#ifdef _OPENMP
            else if constexpr (options::has(s, options::omp, options::omp_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return omp::task_dep[inner_settings{}](tree, fmmoperators, op);
            }
#endif
            else if constexpr (options::has(s, options::timit))
            {
                return sequential::sequential[s](tree, fmmoperators, op);
            }
        }
        template<typename... S, typename TreeS, typename TreeT, typename NearField, typename FarField>
        inline auto fmm(options::settings<S...> s, TreeS& tree_source, TreeT& tree_target,
                        operators::fmm_operators<NearField, FarField> const& fmmoperators, unsigned int op = operators_to_proceed::all) -> void
        {
            static_assert(
              options::support(s, options::_s(options::omp, options::omp_timit, options::seq, options::seq_timit)),
              "unsupported fmm algo options!");
            if constexpr(options::has(s, options::seq, options::seq_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return sequential::sequential[inner_settings{}](tree_source, tree_target, fmmoperators, op);
            }
#ifdef _OPENMP
            else if constexpr(options::has(s, options::omp, options::omp_timit))
            {
                using inner_settings = typename decltype(s)::inner_settings;
                return omp::task_dep[inner_settings{}](tree_source, tree_target, fmmoperators, op);
            }
#endif
            else if constexpr(options::has(s, options::timit))
            {
                return sequential::sequential[s](tree_source, tree_target, fmmoperators, op);
            }
        }
    }

    DECLARE_OPTIONED_CALLEE(fmm);
}

#endif // SCALFMM_ALGORITHMS_FMM_HPP
