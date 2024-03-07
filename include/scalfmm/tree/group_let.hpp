#ifndef SCALFMM_TREE_LET_HPP
#define SCALFMM_TREE_LET_HPP
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cpp_tools/parallel_manager/parallel_manager.hpp>
#include <cpp_tools/colors/colorized.hpp>

#include <scalfmm/tree/utils.hpp>
#include <scalfmm/utils/io_helpers.hpp>   // for io::print
#include <scalfmm/utils/math.hpp>

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/parallel/mpi/utils.hpp"
#include "scalfmm/tree/for_each.hpp"

#ifdef SCALFMM_USE_MPI
#include <inria/algorithm/distributed/distribute.hpp>
#include <inria/algorithm/distributed/mpi.hpp>
#include <inria/algorithm/distributed/sort.hpp>
#include <inria/linear_tree/balance_tree.hpp>
#include <mpi.h>
#endif

namespace scalfmm::tree
{
    using morton_type = std::int64_t;   // typename Tree_type::

    template<typename MortonIdx>
    struct leaf_info_type
    {
        using morton_type = MortonIdx;
        MortonIdx morton{};
        std::size_t number_of_particles{};
        friend std::ostream& operator<<(std::ostream& os, const leaf_info_type& w)
        {
            os << "[" << w.morton << ", " << w.number_of_particles << "] ";
            return os;
        }
    };

    namespace let
    {

        ///
        /// \brief  get theoretical p2p interaction list outside me
        ///
        /// We return the list of indexes of cells involved in P2P interaction that we do
        ///  not have locally.  The cells on other processors may not exist.
        ///
        /// \param[in]  para the parallel manager
        /// \param tree the tree used to compute the interaction
        /// \param local_morton_vect the vector of local morton indexes on my node
        /// \param leaves_distrib the leaves distribution on the processes
        /// \return the list of indexes on other processes
        ///

        template<typename MortonIdx, typename MortonDistribution>
        inline bool is_inside_distrib(MortonIdx morton_idx, MortonDistribution const& leaves_distrib)
        {
            for(auto const& interval: leaves_distrib)
            {
                if(morton_idx > interval[1])
                {
                    continue;
                }
                else if(morton_idx >= interval[0])
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            return false;
        }
        template<typename MortonIdx, typename MortonDistribution>
        inline bool is_inside_distrib_right(MortonIdx morton_idx, int const& start,
                                            MortonDistribution const& leaves_distrib)
        {
            for(int i = start + 1; leaves_distrib.size(); ++i)
            {
                auto const& interval = leaves_distrib[i];
                if(morton_idx > interval[1])
                {
                    continue;
                }
                else if(morton_idx >= interval[0])
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            return false;
        }
        template<typename MortonIdx, typename MortonDistribution>
        inline bool is_inside_distrib_left(MortonIdx morton_idx, int const& rank, MortonDistribution const& distrib)
        {
            // if(rank == 2)
            // {
            //     std::cout << "is_inside_distrib_left morton_idx: " << morton_idx << std::endl;
            // }
            for(int i = rank - 1; i >= 0; --i)
            {
                auto const& interval = distrib[i];
                if(math::between(morton_idx, interval[0], interval[1]))
                {
                    return true;
                }
                else if(morton_idx > interval[1])
                {
                    return false;
                }
                // else if(morton_idx >= interval[0])
                // {
                //     return true;
                // }
            }
            return false;
        }
        template<typename Box, typename VectorLeafInfo, typename MortonDistribution>
        inline /*std::vector<morton_type>*/ VectorLeafInfo
        get_ghosts_p2p_interaction(cpp_tools::parallel_manager::parallel_manager& para, Box const& box,
                                   std::size_t const& level, int const& separation, VectorLeafInfo const& leaf_info,
                                   MortonDistribution const& leaves_distrib)
        {
            std::vector<morton_type> ghost_to_add;
            auto const& period = box.get_periodicity();
            const auto rank = para.get_process_id();
            auto const& my_distrib = leaves_distrib[rank];
            //
            for(auto const& info: leaf_info)
            {
                auto const& morton_index = info.morton;
                auto coordinate{index::get_coordinate_from_morton_index<Box::dimension>(morton_index)};
                auto interaction_neighbors = index::get_neighbors(coordinate, level, period, separation);
                auto& list = std::get<0>(interaction_neighbors);
                auto nb = std::get<1>(interaction_neighbors);
                int it{0};
                // io::print("rank(" + std::to_string(rank) + ") list idx(p2p)  : ", list);

                while(list[it] < my_distrib[0])
                {
                    // std::cout << "INSIDE left idx " << list[it] << "  " << std::boolalpha
                    //           << is_inside_distrib(list[it], leaves_distrib) << std::endl;
                    if(is_inside_distrib_left(list[it], rank, leaves_distrib))
                    {
                        ghost_to_add.push_back(list[it]);
                    }
                    ++it;
                }
                it = nb - 1;
                while(list[it] > my_distrib[1])
                {
                    //     std::cout << "INSIDE right idx " << list[it] << "  " << std::boolalpha
                    //               << is_inside_distrib(list[it], leaves_distrib) << std::endl;
                    if(is_inside_distrib_right(list[it], rank, leaves_distrib))
                    {
                        ghost_to_add.push_back(list[it]);
                    }
                    --it;
                }
            }
            std::sort(ghost_to_add.begin(), ghost_to_add.end());
            auto last = std::unique(ghost_to_add.begin(), ghost_to_add.end());
            ghost_to_add.erase(last, ghost_to_add.end());
            VectorLeafInfo ghost_leaf_to_add(ghost_to_add.size());
            for(int i = 0; i < ghost_to_add.size(); ++i)
            {
                ghost_leaf_to_add[i] = {ghost_to_add[i], 0};
            }

            return ghost_leaf_to_add;
        }
        ///
        /// \brief  get theoretical m2l interaction list outside me
        ///
        /// We return the list of indexes of cells involved in P2P interaction that we do
        ///  not have locally.  The cells on other processors may not exist.
        ///
        /// \param[in]  para the parallel manager
        /// \param tree the tree used to compute the interaction
        /// \param local_morton_idx the local morton index of the cells
        /// \param cell_distrib the cells distribution on the processes
        /// \return the list of indexes on tother processes
        ///
        template<typename Box, typename VectorMortonIdx, typename MortonDistribution>
        inline VectorMortonIdx
        get_ghosts_m2l_interaction(cpp_tools::parallel_manager::parallel_manager& para, Box const& box,
                                   const std::size_t& level, int const& separation,
                                   VectorMortonIdx const& local_morton_vect, MortonDistribution const& cell_distrib)
        {
            VectorMortonIdx ghost_to_add;
            auto const& period = box.get_periodicity();
            const auto rank = para.get_process_id();
            auto const my_distrib = cell_distrib[rank];
            //
            for(auto morton_index: local_morton_vect)
            {
                auto coordinate{index::get_coordinate_from_morton_index<Box::dimension>(morton_index)};
                auto interaction_m2l_list = index::get_m2l_list(coordinate, level, period, separation);
                auto& list = std::get<0>(interaction_m2l_list);
                auto nb = std::get<2>(interaction_m2l_list);
                //
                // io::print("rank(" + std::to_string(rank) + ") list idx(m2l)  : ", list);
                // io::print("rank(" + std::to_string(rank) + ") cell_distrib  : ", cell_distrib);

                int it{0};
                for(auto it = 0; it < nb; ++it)
                {
                    if(list[it] > my_distrib[0])
                    {
                        break;
                    }
                    bool check{false};
                    // for(int i = 0; i < rank; ++i)
                    for(int i = rank - 1; i >= 0; i--)
                    {
                        auto const& interval = cell_distrib[i];
                        // // if(rank == 2)
                        // {
                        //     std::cout << "is_inside_distrib_left list[it]: " << interval[0] << " < " << list[it]
                        //               << " < " << interval[1] << std::endl;
                        // }
                        check = math::between(list[it], interval[0], interval[1]);
                        if(check)
                        {
                            break;
                        }
                    }
                    // std::cout << "                 " << list[it] << "  " << std::boolalpha << check << std::endl;
                    if(check)   // is_inside_distrib_left(list[it], rank, cell_distrib))
                    {
                        ghost_to_add.push_back(list[it]);
                    }
                }
                // while(list[it] < my_distrib[0])
                // {
                //     std::cout << it << " INSIDE left idx " << list[it] << "  " << std::boolalpha
                //               << is_inside_distrib(list[it], cell_distrib) << std::endl;
                //     if(is_inside_distrib_left(list[it], rank, cell_distrib))
                //     {
                //         ghost_to_add.push_back(list[it]);
                //     }
                //     ++it;
                //     if(it > nb)
                //     {
                //         break;
                //     }
                // }
                it = nb - 1;
                while(list[it] > my_distrib[1])
                {
                    if(is_inside_distrib_right(list[it], rank, cell_distrib))
                    {
                        ghost_to_add.push_back(list[it]);
                    }
                    --it;
                    // ghost_to_add.push_back(list[it]);
                    // --it;
                }
                // if(rank == 2)
                // {
                //     io::print("rank(" + std::to_string(rank) + ") tmp ghost_to_add(m2l)  : ", ghost_to_add);
                // }
            }
            std::sort(ghost_to_add.begin(), ghost_to_add.end());
            auto last = std::unique(ghost_to_add.begin(), ghost_to_add.end());
            ghost_to_add.erase(last, ghost_to_add.end());
            // io::print("rank(" + std::to_string(rank) + ") ghost_to_add(m2l)  : ", ghost_to_add);

            return ghost_to_add;
        }

        /**
         * @brief Merge the P2P ghosts with the M2L ghosts
         *
         * @tparam VectorLeafInfo is a vector of leafinfo (morton, number_of_particles)
         * @tparam VectorMortonIdx
         * @param ghostP2P_leafInfo
         * @param ghost_m2l_cells
         * @return VectorLeafInfo
         */
        template<typename VectorLeafInfo, typename VectorMortonIdx>
        VectorLeafInfo merge_leaves_cells(VectorLeafInfo& ghostP2P_leafInfo, VectorMortonIdx& ghost_m2l_cells)
        {
            // compute the size of the merged vector
            std::size_t i{0}, j{0}, k{0};
            std::size_t size1{ghostP2P_leafInfo.size()}, size2{ghost_m2l_cells.size()};
            while(i < size1 && j < size2)
            {
                if(ghostP2P_leafInfo[i].morton < ghost_m2l_cells[j])
                {
                    ++i;
                }
                else if(ghostP2P_leafInfo[i].morton > ghost_m2l_cells[j])
                {
                    ++j;
                }
                else
                {
                    ++i;
                    ++j;
                }
                ++k;
            }

            VectorLeafInfo ghosts(k + size1 - i + size2 - j);
            i = j = k = 0;
            while(i < size1 && j < size2)
            {
                if(ghostP2P_leafInfo[i].morton < ghost_m2l_cells[j])
                {
                    ghosts[k++] = ghostP2P_leafInfo[i++];
                }
                else if(ghostP2P_leafInfo[i].morton > ghost_m2l_cells[j])
                {
                    ghosts[k++].morton = ghost_m2l_cells[j++];
                }
                else
                {
                    ghosts[k++] = ghostP2P_leafInfo[i++];
                    j++;
                }
            }

            // Add the remaining elements of vector ghostP2P_leafInfo (if any)
            while(i < size1)
            {
                ghosts[k++] = ghostP2P_leafInfo[i++];
            }

            // Add the remaining elements of vector ghost_m2l_cells (if any)
            while(j < size2)
            {
                ghosts[k++].morton = ghost_m2l_cells[j++];
            }

            return ghosts;
        }
        template<typename VectorLeafInfoType>
        auto merge_split_structure(VectorLeafInfoType const& localLeaves, VectorLeafInfoType const& ghosts)
        {
            // compute the size of the merged vector
            using morton_type = typename VectorLeafInfoType::value_type::morton_type;
            std::size_t i{0}, j{0}, k{0};
            std::size_t size1{localLeaves.size()}, size2{ghosts.size()};

            std::vector<morton_type> morton(size1 + size2);
            std::vector<std::size_t> number_of_particles(size1 + size2);
            i = j = k = 0;
            while(i < size1 && j < size2)
            {
                if(localLeaves[i].morton < ghosts[j].morton)
                {
                    morton[k] = localLeaves[i].morton;
                    number_of_particles[k++] = localLeaves[i++].number_of_particles;
                }
                else if(localLeaves[i].morton > ghosts[j].morton)
                {
                    morton[k] = ghosts[j].morton;
                    number_of_particles[k++] = ghosts[j++].number_of_particles;
                }
                else
                {
                    morton[k] = localLeaves[i].morton;
                    number_of_particles[k++] = localLeaves[i++].number_of_particles;
                    j++;
                }
            }

            // Add the remaining elements of vector localLeaves (if any)
            while(i < size1)
            {
                morton[k] = localLeaves[i].morton;
                number_of_particles[k++] = localLeaves[i++].number_of_particles;
            }

            // Add the remaining elements of vector ghost_m2l_cells (if any)
            while(j < size2)
            {
                morton[k] = ghosts[j].morton;
                number_of_particles[k++] = ghosts[j++].number_of_particles;
            }

            return std::make_tuple(morton, number_of_particles);
        }
        /**
         * @brief Split the LeafInfo structure in two vectors (Morton, number_of_particles)
         *
         * @tparam VectorLeafInfoType
         * @param leaves
         * @return a tuple of two vectors the morton index and the number of particles in the leaves vector
         */
        template<typename VectorLeafInfoType>
        auto split_structure(VectorLeafInfoType const& leaves)
        {
            // compute the size of the merged vector
            using morton_type = typename VectorLeafInfoType::value_type::morton_type;
            std::size_t k{0};
            std::size_t size{leaves.size()};

            std::vector<morton_type> morton(size);
            std::vector<std::size_t> number_of_particles(size);
            for(auto& v: leaves)
            {
                morton[k] = v.morton;
                number_of_particles[k++] = v.number_of_particles;
            }
            return std::make_tuple(morton, number_of_particles);
        }
        template<typename VectorLeafInfoIteratorType>
        auto split_structure(const VectorLeafInfoIteratorType begin, const VectorLeafInfoIteratorType end)
        {
            // compute the size of the merged vector
            using VectorLeafInfoType = std::decay_t<decltype(*begin)>;
            using morton_type = typename VectorLeafInfoType::morton_type;
            std::size_t k{0};
            auto size{std::distance(begin, end)};
            std::vector<morton_type> morton(size);
            std::vector<std::size_t> number_of_particles(size);
            for(auto it = begin; it != end; ++it)
            {
                morton[k] = (*it).morton;
                number_of_particles[k++] = (*it).number_of_particles;
            }
            return std::make_tuple(morton, number_of_particles);
        }
        // template<typename VectorMorton>
        // auto get_sub_vector(const VectorMorton begin, const VectorMorton end)
        // {
        //     // compute the size of the merged vector
        //     using morton_type = std::decay_t<decltype(*begin)>;

        //     std::size_t k{0};
        //     auto size{std::distance(begin, end)};
        //     std::vector<morton_type> morton(size);
        //     for(auto it = begin; it != end; ++it)
        //     {
        //         morton[k] = (*it);
        //     }
        //     return morton;
        // }
        /**
         * @brief
         *
         * @tparam VectorBlockType
         * @param bloc1
         * @param bloc2
         * @param bloc3
         * @return VectorBlockType
         */
        template<typename VectorBlockType>
        VectorBlockType merge_blocs(VectorBlockType const& bloc1, VectorBlockType const& bloc2,
                                    VectorBlockType const& bloc3)
        {
            // Merge the three block structure
            auto size = bloc1.size() + bloc2.size() + bloc3.size();
            std::cout << " all size: " << size << std::endl;
            VectorBlockType all_blocks(size);
            int k{0};
            for(int i = 0; i < bloc1.size(); ++i)
            {
                all_blocks[k++] = bloc1[i];
            }
            for(int i = 0; i < bloc2.size(); ++i)
            {
                all_blocks[k++] = bloc2[i];
            }
            for(int i = 0; i < bloc3.size(); ++i)
            {
                all_blocks[k++] = bloc3[i];
            }
            return all_blocks;
        }
        ///
        /// \brief construct the local essential tree (LET) at the level.
        ///
        ///  We start from a given Morton index distribution and we compute all
        ///  interactions needed
        ///   in the algorithm steps.
        ///  At the leaf level it corresponds to the interactions coming from the
        ///  direct pass (P2P operators)
        ///     and in the transfer pass (M2L operator). For the other levels we
        ///     consider only the M2L interactions.
        /// The leaves_distrib and the cells_distrib might be different
        ///  At the end the let has also all the interaction list computed
        ///
        /// \param[inout]  tree the tree to compute the let.
        /// \param[in]  local_morton_idx the morton index of the particles in the
        /// processors.
        ///
        ///
        ///  \param[in]  cells_distrib the morton index distribution for
        /// the cells at the leaf level.
        ///
        ///  \param[in]  level the level to construct the let
        ///
        template<typename Box, typename VectorMortonIdx, typename MortonDistribution>
        [[nodiscard]] auto build_let_at_level(cpp_tools::parallel_manager::parallel_manager& para, Box& box,
                                              const int& level, const VectorMortonIdx& local_morton_vect,
                                              const MortonDistribution& cells_distrib, const int& separation)
          -> VectorMortonIdx
        {
            const auto my_rank = para.get_process_id();
            // std::cout << cpp_tools::colors::red << " --> Begin let::build_let_at_level() at level = " << level
            //           << "dist: " << cells_distrib[my_rank] << cpp_tools::colors::reset << std::endl;
            // io::print("rank(" + std::to_string(my_rank) + ") local_morton_vect  : ", local_morton_vect);

            //  we compute the cells needed in the M2L operator
            auto needed_idx =
              std::move(get_ghosts_m2l_interaction(para, box, level, separation, local_morton_vect, cells_distrib));

            // io::print("rank(" + std::to_string(my_rank) + ") needed_idx(m2l)  : ", needed_idx);

            std::cout << std::flush;
            /// Look if the morton index really exists in the distributed tree

            parallel::utils::check_if_morton_index_exist(para, needed_idx, cells_distrib, local_morton_vect);
            ///
            // io::print("rank(" + std::to_string(my_rank) + ") final idx(m2l)  : ", needed_idx);
            //

            // std::cout << cpp_tools::colors::red
            //           << "rank(" + std::to_string(my_rank) + ")-- > End let::build_let_at_level() "
            //           << cpp_tools::colors::reset << std::endl;
            return needed_idx;
        }
        // template<typename OctreeTree, typename VectorMortonIdx, typename MortonDistribution>
        // void build_let_at_level(cpp_tools::parallel_manager::parallel_manager& para, OctreeTree& tree,
        //                         const VectorMortonIdx& local_morton_idx, const MortonDistribution& cells_distrib,
        //                         const int& level)
        // {
        //     std::cout << cpp_tools::colors::green << " --> Begin let::build_let_at_level() at level = " << level
        //               << cpp_tools::colors::reset << std::endl;

        //     // auto my_rank = para.get_process_id();
        //     // // stock in the variable if we are at the leaf level
        //     // bool leaf_level = (tree.leaf_level() == level);

        //     // //  we compute the cells needed in the M2L operators
        //     // auto needed_idx =
        //     //   std::move(distrib::get_m2l_interaction_at_level(para, tree, local_morton_idx, cells_distrib,
        //     //   level));
        //     // //           io::print("rank(" + std::to_string(my_rank) + ") needed_idx(m2l)  : ", needed_idx);
        //     // // std::cout << std::flush;
        //     // /// Look if the morton index really exists in the distributed tree
        //     // distrib::check_if_morton_index_exist(para, needed_idx, cells_distrib, local_morton_idx);
        //     // //            io::print("rank(" + std::to_string(my_rank) + ") final idx(m2l)  : ", needed_idx);
        //     // //            std::cout << std::flush;
        //     // ///
        //     // tree.insert_cells_at_level(level, needed_idx);
        //     std::cout << cpp_tools::colors::green << " --> End let::build_let_at_level() at level = " << level
        //               << cpp_tools::colors::reset << std::endl;
        // }
        /**
         * @brief
         *
         * @tparam Box
         * @tparam VectorMortonIdx
         * @tparam MortonDistribution
         * @param para
         * @param box
         * @param level
         * @param leaf_info
         * @param leaves_distrib
         * @param separation
         */
        template<typename Box, typename VectorLeafInfo, typename MortonDistribution>
        [[nodiscard]] auto build_let_leaves(cpp_tools::parallel_manager::parallel_manager& para, Box const& box,
                                            const std::size_t& level,
                                            const VectorLeafInfo& leaf_info /*local_morton_vect*/,
                                            MortonDistribution const& leaves_distrib, const int& separation)

          -> VectorLeafInfo
        {
            auto my_rank = para.get_process_id();
            // std::cout << cpp_tools::colors::green
            //           << "rank(" + std::to_string(my_rank) + ") --> Begin let::build_let_leaves() "
            //           << cpp_tools::colors::reset << std::endl;
            // io::print("rank(" + std::to_string(my_rank) + ") leaf_info  : ", leaf_info);

            //  we compute the leaves involved in the P2P operators
            auto leaf_info_to_add =
              std::move(get_ghosts_p2p_interaction(para, box, level, separation, leaf_info, leaves_distrib));
            // io::print("rank(" + std::to_string(my_rank) + ") leaf_info_to_add(p2p)  : ", leaf_info_to_add);

            std::vector<morton_type> needed_idx(leaf_info_to_add.size());
            for(int i = 0; i < leaf_info_to_add.size(); ++i)
            {
                needed_idx[i] = leaf_info_to_add[i].morton;
            }
            /// Look if the morton index really exists in the distributed tree
            /// needed_idx input  contains the Morton index of leaf
            ///            output   contains the number of particles in the leaf
            parallel::utils::check_if_leaf_morton_index_exist(para, needed_idx, leaves_distrib, leaf_info);
            int idx{0};

            for(int i = 0; i < needed_idx.size(); ++i)
            {
                if(needed_idx[i] > 0)
                {
                    leaf_info_to_add[idx].morton = leaf_info_to_add[i].morton;
                    leaf_info_to_add[idx].number_of_particles = needed_idx[i];
                    ++idx;
                }
            }
            if(idx != needed_idx.size())
            {
                auto last = leaf_info_to_add.cbegin() + idx;
                leaf_info_to_add.erase(last, leaf_info_to_add.end());
            }
            ///
            // io::print("rank(" + std::to_string(my_rank) + ") final leaf_info_to_add(p2p)  : ", leaf_info_to_add);
            // std::cout << cpp_tools::colors::green
            //           << "rank(" + std::to_string(my_rank) + ")-- > End let::build_let_leaves() "
            //           << cpp_tools::colors::reset << std::endl;
            return leaf_info_to_add;
        }

        ///
        /// \brief buildLetTree  Build the let of the tree and the leaves and cells distributions
        ///
        /// The algorithm has 5 steps:
        ///   1) We sort the particles according to their Morton Index (leaf level)
        ///   2) Build the leaf morton vector of my local particles and construct either
        ///      the leaves distribution or the cell distribution according to parameter
        ///       use_leaf_distribution or use_particle_distribution
        ///   3) Fit the particles inside the use_leaf_distribution
        ///   4) Construct the  tree according to my particles and build the leaf
        ///       morton vector of my local particles
        ///   5) Constructing the let level by level
        ///
        /// \param[in]    manager   the parallel manager
        /// \param[in] number_of_particles  total number of particles in the simulation
        /// \param[in]  particle_container   vector of particles on my node. On output the
        ///                 array is sorted and correspond to teh distribution built
        /// \param[in]     box  size of the simulation box
        /// \param[in] leaf_level   level of the leaf in the tree
        /// \param[in] groupSizeLeaves  blocking parameter for the leaves (particles)
        /// \param[in] groupSizeCells    blocking parameter for the cells
        /// @param[in]    order order of the approximation to build the tree
        /// @param[in]    use_leaf_distribution to say if you consider the leaf distribution
        /// @param[in]    use_particle_distribution to say if you consider the particle distribution
        /// @return  localGroupTree  the LET of the octree

        /// processors
        template<typename Tree_type, typename Vector_type, typename Box_type>
        Tree_type buildLetTree(cpp_tools::parallel_manager::parallel_manager& manager,
                               const std::size_t& number_of_particles, Vector_type& particle_container,
                               const Box_type& box, const int& leaf_level, const int groupSizeLeaves,
                               const int groupSizeCells, const int order, const int separation,
                               const bool use_leaf_distribution, const bool use_particle_distribution)
        {
            std::cout << cpp_tools::colors::green << " --> Begin let::group_let() " << cpp_tools::colors::reset
                      << std::endl;
            //
            static constexpr std::size_t dimension = Vector_type::value_type::dimension;
            const auto rank = manager.get_process_id();
            auto nb_part = particle_container.size();
            ////////////////////////////////////////////////////////////////////////////
            ///   Sort the particles at the leaf level according to their Morton index
#ifdef SCALFMM_USE_MPI
            inria::mpi_config conf_tmp(manager.get_communicator().raw_comm);

            inria::sort(conf_tmp.comm, particle_container,
                        [&box, &leaf_level](const auto& p1, const auto& p2)
                        {
                            auto m1 = scalfmm::index::get_morton_index(p1.position(), box, leaf_level);
                            auto m2 = scalfmm::index::get_morton_index(p2.position(), box, leaf_level);
                            return m1 < m2;
                        });
#else
            std::sort(particle_container.begin(), particle_container.end(),
                      [&box, &leaf_level](const auto& p1, const auto& p2) {
                          auto m1 = scalfmm::index::get_morton_index(p1.position(), box, leaf_level);
                          auto m2 = scalfmm::index::get_morton_index(p2.position(), box, leaf_level);
                          return m1 < m2;
                      });
#endif
            // Build the morton index of the particles in order to find the
            //  existing leaves
            const std::size_t localNumberOfParticles = particle_container.size();
            std::vector<morton_type> particleMortonIndex(localNumberOfParticles);
            // As the particles are sorted the leafMortonIdx is sorted too
#pragma omp parallel for shared(localNumberOfParticles, box, leaf_level)
            for(std::size_t part = 0; part < localNumberOfParticles; ++part)
            {
                particleMortonIndex[part] =
                  scalfmm::index::get_morton_index(particle_container[part].position(), box, leaf_level);
            }
            auto leafMortonIdx(particleMortonIndex);
            // delete duplicate indexes
            auto last = std::unique(leafMortonIdx.begin(), leafMortonIdx.end());
            leafMortonIdx.erase(last, leafMortonIdx.end());
            ///////////////////////////////////////////////////////////////////////////////////
            io::print("rank(" + std::to_string(rank) + ")  -->  init leafMortonIdx: ", leafMortonIdx);
            ///
            ////////////////////////////////////////////////////////////////////////////////////////////
            ////   Construct a uniform distribution for the leaves/cells at the leaves level
            ///
            /// A morton index should be own by only one process
            ///
            using morton_distrib_type = typename Tree_type::data_distrib_type;

            ///
            ///  Build a uniform distribution of the leaves/cells
            ///
            morton_distrib_type leaves_distrib;
            if(use_leaf_distribution)
            {
                leaves_distrib = std::move(scalfmm::parallel::utils::balanced_leaves(manager, leafMortonIdx));
            }
            io::print("rank(" + std::to_string(rank) + ")  -->  leaves_distrib: ", leaves_distrib);
            ////                End
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////
            ////   Construct a uniform distribution for the particles
            ///  On each process we have the same number of particles. The number of leaves might differ significally
            ///
            /// A morton index should be own by only one process
            ///
            morton_distrib_type particles_distrib(manager.get_num_processes());
            if(use_particle_distribution)
            {
                particles_distrib = std::move(scalfmm::parallel::utils::balanced_particles(
                  manager, particle_container, particleMortonIndex, number_of_particles));
                if(!use_leaf_distribution)
                {
                    leaves_distrib.resize(particles_distrib.size());
                    std::copy(particles_distrib.begin(), particles_distrib.end(), leaves_distrib.begin());
                }
            }
            else
            {
                particles_distrib = leaves_distrib;
            }
            if(manager.io_master())
            {
                std::cout << cpp_tools::colors::red;
                io::print("rank(" + std::to_string(rank) + ")  -->  particles_distrib: ", particles_distrib);
                io::print("rank(" + std::to_string(rank) + ")  -->  leaves_distrib:    ", leaves_distrib);
                std::cout << cpp_tools::colors::reset << std::endl;
            }
            ////                End
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            /// Check the two distributions
            ///
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///   Set the particles on the good process according to the computed distribution
            ///
            parallel::utils::fit_particles_in_distrib(manager, particle_container, particleMortonIndex,
                                                      particles_distrib, box, leaf_level, number_of_particles);
            ///    All the particles are located on the good process
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///   Construct the local tree based on our set of particles
            // Build and empty tree
            Tree_type localGroupTree(static_cast<std::size_t>(leaf_level + 1), order, groupSizeLeaves, groupSizeCells,
                                     box);
            /// Set true because the particles are already sorted
            ///  In fact we have all the leaves to add in leafMortonIdx - could be used to construct
            /// the tree !!!
            ///

#ifdef SCALFMM_USE_MPI

            // std::cout << cpp_tools::colors::red;
            // io::print("rank(" + std::to_string(rank) + ") leafMortonIdx: ", leafMortonIdx);
            // std::cout << cpp_tools::colors::reset << std::endl;
            ///  End
            ////////////////////////////////////////////////////////////////////////////////////////////
            ///
            /// Compute the new morton indexes associated to the particles on the process
            ///
            leafMortonIdx.resize(particle_container.size());
#pragma omp parallel for shared(localNumberOfParticles, box, leaf_level)
            for(std::size_t i = 0; i < particle_container.size(); ++i)
            {
                leafMortonIdx[i] = scalfmm::index::get_morton_index(particle_container[i].position(), box, leaf_level);
            }
            // localLeafInfo contains information on leaves (morton, number of particles) own by th current process
            std::vector<tree::leaf_info_type<morton_type>> localLeafInfo(leafMortonIdx.size());
            auto start{leafMortonIdx[0]};
            int idx{0};
            localLeafInfo[idx].number_of_particles = 1;
            localLeafInfo[idx].morton = start;
            for(std::size_t i = 1; i < particle_container.size(); ++i)
            {
                if(leafMortonIdx[i] == start)
                {
                    localLeafInfo[idx].number_of_particles += 1;
                }
                else
                {
                    idx++;
                    start = leafMortonIdx[i];
                    localLeafInfo[idx].number_of_particles = 1;
                    localLeafInfo[idx].morton = start;
                    leafMortonIdx[idx] = leafMortonIdx[i];
                }
            }
            leafMortonIdx.resize(idx + 1);
            localLeafInfo.resize(leafMortonIdx.size());
            io::print("rank(" + std::to_string(rank) + ")  -->  localLeafInfo:    ", localLeafInfo);
            io::print("rank(" + std::to_string(rank) + ")  -->  leafMortonIdx:    ", leafMortonIdx);
            ////////////////////////////////////////////////////////////////////////////////////////
            // Build the pointer of the tree with all parameters

            if(manager.get_num_processes() > 1)
            {
                ////////////////////////////////////////////////////////////////////////////////////////////
                ///   Step 5    Construct the let according to the distributions particles and cells
                ///
                /// Find and add the leaves to add at the leaves level
                ///   we consider the particles_distrib

                auto ghostP2P_leafInfo =
                  build_let_leaves(manager, box, leaf_level, localLeafInfo, particles_distrib, separation);
                io::print("rank(" + std::to_string(rank) + ")  -->  final ghostP2P_leafInfo:    ", ghostP2P_leafInfo);
                io::print("rank(" + std::to_string(rank) + ")  -->  final localLeafInfo:    ", localLeafInfo);

                localGroupTree.set_leaf_distribution(particles_distrib);

                // std::cout << std::flush;
                // std::cout << cpp_tools::colors::red;
                // std::cout << "END LEAF LEVEL " << std::endl;
                // std::cout << cpp_tools::colors::reset;

                /// If the distribution is not the same for the leaf and the cell we redistribute the
                /// morton index according to the uniform distribution of morton index
                ///
                //////////////////////////////////////////////////////////////////
                ///  Construct a  uniform distribution of the morton index
                ///
#ifdef TEST_
                if(use_leaf_distribution && use_particle_distribution)
                {
                    std::cout << cpp_tools::colors::red << "WARNING\n" << cpp_tools::colors::reset << std::endl;
                    try
                    {
                        inria::mpi_config conf_tmp(manager.get_communicator().raw_comm);
                        inria::distribute(conf_tmp, leafMortonIdx,
                                          inria::uniform_distribution{conf_tmp, leafMortonIdx});
                    }
                    catch(std::out_of_range& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                }
#endif
                // ///
                // /// Find and add the cells to add at the leaves level
                std::vector<morton_distrib_type> level_dist(localGroupTree.height());
                level_dist[leaf_level] = leaves_distrib;
                localGroupTree.set_cell_distribution(leaf_level, level_dist[leaf_level]);

                auto ghost_m2l_cells =
                  build_let_at_level(manager, box, leaf_level, leafMortonIdx, level_dist[leaf_level], separation);
                io::print("rank(" + std::to_string(rank) + ")  -->  final ghost_cells(m2l):    ", ghost_m2l_cells);

                // distribution, particles
                localGroupTree.create_from_leaf_level(localLeafInfo, ghostP2P_leafInfo, ghost_m2l_cells,
                                                      particles_distrib[rank], level_dist[leaf_level][rank]);

                // build all leaves at leaf level
                for(int level = leaf_level - 1; level >= localGroupTree.top_level(); --level)
                {
                    level_dist[level] = std::move(parallel::utils::build_upper_distribution(
                      manager, dimension, level, leafMortonIdx, level_dist[level + 1]));
                    //                    io::print("rank(" + std::to_string(rank) + ") leafMortonIdx: ",
                    //                    leafMortonIdx); io::print("rank(" + std::to_string(rank) + ") part: ",
                    //                    level_dist[l]);
                    localGroupTree.set_cell_distribution(level, level_dist[level]);
                    auto ghost_cells_level =
                      build_let_at_level(manager, box, level, leafMortonIdx, level_dist[level], separation);
                    io::print("rank(" + std::to_string(rank) + ") level=" + std::to_string(level) +
                                " -->  final ghost_cells(m2l):    ",
                              ghost_cells_level);
                    localGroupTree.create_cells_at_level(level, leafMortonIdx, ghost_cells_level,
                                                         level_dist[level][rank]);
                }
                // Il faut construire la numerotation globale des groupes
                // Mpi_sum_prefix ?
                manager.get_communicator().barrier();
                std::cout << "   END FIRST PART\n" << std::flush;
            }
            else
#endif
            {
                localGroupTree.set_leaf_distribution(particles_distrib);
                localGroupTree.set_cell_distribution(leaf_level, leaves_distrib);

                for(int l = leaf_level - 1; l >= localGroupTree.top_level(); --l)
                {
                    leaves_distrib[0][0] = leaves_distrib[0][0] >> dimension;
                    leaves_distrib[0][1] = leaves_distrib[0][1] >> dimension;
                    localGroupTree.set_cell_distribution(l, leaves_distrib);
                }
            }

            std::cout << cpp_tools::colors::red << std::endl;
            std::cout << "set iterators \n";
            localGroupTree.set_valid_iterators();
            std::cout << "begin fill_leaves_with_particles \n";
            localGroupTree.fill_leaves_with_particles(particle_container);
            std::cout << "end fill_leaves_with_particles \n";

            std::cout << cpp_tools::colors::reset << std::endl;
            std::cout << cpp_tools::colors::green << " --> End let::group_let() " << cpp_tools::colors::reset
                      << std::endl;

            return localGroupTree;
        }

    }   // namespace let
}   // namespace scalfmm::tree

#endif
