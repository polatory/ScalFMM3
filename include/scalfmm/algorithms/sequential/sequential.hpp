// --------------------------------
// See LICENCE file at project root
// File : algorithm/sequential/sequential.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_SEQUENTIAL_SEQUENTIAL_HPP
#define SCALFMM_ALGORITHMS_SEQUENTIAL_SEQUENTIAL_HPP
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

#include <cpp_tools/timers/simple_timer.hpp>

#include "scalfmm/algorithms/common.hpp"
#include "scalfmm/algorithms/sequential/cell_to_leaf.hpp"
#include "scalfmm/algorithms/sequential/direct.hpp"
#include "scalfmm/algorithms/sequential/downward.hpp"
#include "scalfmm/algorithms/sequential/leaf_to_cell.hpp"
#include "scalfmm/algorithms/sequential/periodic.hpp"
#include "scalfmm/algorithms/sequential/transfer.hpp"
#include "scalfmm/algorithms/sequential/upward.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/tools/bench.hpp"

namespace scalfmm::algorithms::sequential
{
    namespace impl
    {
        /** \brief Sequential algorithm
         *
         * This function launches all the passes of the algorithm.
         *
         * \tparam TreeIterator an iterator on the beginning of the tree
         * \tparam Interpolator an iterator on the beginning of the tree
         * \param begin
         * \param end
         * \param op an int
         * \return void
         */

        /** \brief Sequential algorithm
         *
         * This function launches all the passes of the algorithm.
         *
         * \tparam Tree the template og the tree
         * \tparam FmmOperators a fmm operator containing near-field and far-field operators
         * \param tree
         * \param fmmoperators
         * \param op an int
         * \return void
         */
        template<typename TreeSource, typename TreeTarget, typename NearField, typename FarField, typename... S>
        inline auto sequential(options::settings<S...> s, TreeSource& tree_source, TreeTarget& tree_target,
                               operators::fmm_operators<NearField, FarField> const& fmmoperators,
                               unsigned int op_in = operators_to_proceed::all) -> void
        {
            static_assert(options::support(s, options::_s(options::timit)), "sequential algo unsupported options!");
            bool same_tree{false};
            if constexpr(std::is_same_v<TreeSource, TreeTarget>)
            {
                same_tree = (&tree_source == &tree_target);
            }

            using timer_type = cpp_tools::timers::timer<std::chrono::milliseconds>;
            std::unordered_map<std::string, timer_type> timers = {{"p2m", timer_type()}, {"m2m", timer_type()},
                                                                  {"m2l", timer_type()}, {"l2l", timer_type()},
                                                                  {"l2p", timer_type()}, {"p2p", timer_type()}};
            auto const& approximation = fmmoperators.far_field().approximation();
            auto const& neighbour_separation = fmmoperators.near_field().separation_criterion();
            auto const& mutual = fmmoperators.near_field().mutual();

            if(tree_target.is_interaction_m2l_lists_built() == false)
            {
                scalfmm::list::sequential::build_m2l_interaction_list(tree_source, tree_target, neighbour_separation);
            }
            const auto op = tree_target.height() == 2 ? operators_to_proceed::p2p : op_in;

            if((op & operators_to_proceed::p2m) == operators_to_proceed::p2m)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["p2m"].tic();
                }
                pass::leaf_to_cell(tree_source, fmmoperators.far_field());
                if constexpr(options::has(s, options::timit))
                {
                    timers["p2m"].tac();
                }
            }

            if((op & operators_to_proceed::m2m) == operators_to_proceed::m2m)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2m"].tic();
                }
                pass::upward(tree_source, approximation);
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2m"].tac();
                }
            }

            if(same_tree && tree_source.box().is_periodic())
            {
                //  timers["field0"].tic();
                pass::build_field_level0(tree_source, tree_target, approximation);
                //  timers["field0"].tac();
            }
            if((op & operators_to_proceed::m2l) == operators_to_proceed::m2l)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2l"].tic();
                }
                pass::transfer(tree_source, tree_target, fmmoperators.far_field());
                if constexpr(options::has(s, options::timit))
                {
                    timers["m2l"].tac();
                }
            }

            if((op & operators_to_proceed::l2l) == operators_to_proceed::l2l)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2l"].tic();
                }
                pass::downward(tree_target, approximation);
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2l"].tac();
                }
            }

            if((op & operators_to_proceed::l2p) == operators_to_proceed::l2p)
            {
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2p"].tic();
                }
                pass::cell_to_leaf(tree_target, fmmoperators);
                if constexpr(options::has(s, options::timit))
                {
                    timers["l2p"].tac();
                }
            }

            if((op & operators_to_proceed::p2p) == operators_to_proceed::p2p)
            {
                if(tree_target.is_interaction_p2p_lists_built() == false)
                {
                    scalfmm::list::sequential::build_p2p_interaction_list(tree_source, tree_target,
                                                                          neighbour_separation, mutual);
                }

                if constexpr(options::has(s, options::timit))
                {
                    timers["p2p"].tic();
                }
                pass::direct(tree_source, tree_target, fmmoperators.near_field());
                if constexpr(options::has(s, options::timit))
                {
                    timers["p2p"].tac();
                }
            }

            if constexpr(options::has(s, options::timit))
            {
                auto [fartime, neartime, overall, ratio] = bench::print(timers);
                // bench::dump_csv( "timings.csv"
                //                , "groupsize,legende,time,type"
                //                , std::to_string(tree.group_of_leaf_size())
                //                , std::string("far")
                //                , std::to_string(fartime)
                //                , std::string("experimental")
                //                );
                /**/
            }
        }

        template<typename TreeS, typename TreeT, typename NearField, typename FarField>
        inline auto sequential(TreeS& tree_source, TreeT& tree_target,
                               operators::fmm_operators<NearField, FarField> const& fmmoperators,
                               unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential(options::settings<>{}, tree_source, tree_target, fmmoperators, op);
        }
        template<typename Tree, typename NearField, typename FarField>
        inline auto sequential(Tree& tree, operators::fmm_operators<NearField, FarField> const& fmmoperators,
                               unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential(options::settings<>{}, tree, tree, fmmoperators, op);
        }
        template<typename... S, typename Tree, typename NearField, typename FarField>
        inline auto sequential(options::settings<S...> s, Tree& tree,
                               operators::fmm_operators<NearField, FarField> const& fmmoperators,
                               unsigned int op = operators_to_proceed::all) -> void
        {
            return sequential(s, tree, tree, fmmoperators, op);
        }

    }   // namespace impl

    DECLARE_OPTIONED_CALLEE(sequential);

}   // namespace scalfmm::algorithms::sequential

#endif   // SCALFMM_TREE_TREE_HPP
