// --------------------------------
// See LICENCE file at project root
// File : group_tree.hpp
// --------------------------------
#ifndef SCALFMM_TREE_DIST_GROUP_TREE_HPP
#define SCALFMM_TREE_DIST_GROUP_TREE_HPP
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "scalfmm/tree/box.hpp"
#include <scalfmm/tree/group_let.hpp>
#include <scalfmm/tree/group_tree_view.hpp>

#include <scalfmm/utils/io_helpers.hpp>

#include <cpp_tools/colors/colorized.hpp>

namespace scalfmm::component
{
    template<typename Cell, typename Leaf, typename Box = box<typename Leaf::position_type>>
    class dist_group_tree : public group_tree_view<Cell, Leaf, Box>
    {
      public:
        using morton_type = std::int64_t;
        using data_distrib_value_type = std::array<morton_type, 2>;
        using data_distrib_type = std::vector<data_distrib_value_type>;
        using base_type = group_tree_view<Cell, Leaf, Box>;
        using leaf_iterator_type = typename base_type::leaf_iterator_type;
        using const_leaf_iterator_type = typename base_type::const_leaf_iterator_type;
        using cell_group_level_iterator_type = typename base_type::cell_group_level_type::iterator;
        using iterator_type = typename base_type::iterator_type;
        using const_iterator_type = typename base_type::const_iterator_type;
        /// Constructor
        explicit dist_group_tree(std::size_t tree_height, std::size_t order, std::size_t size_leaf_blocking,
                                 std::size_t size_cell_blocking, Box const& box)
          : base_type(tree_height, order, size_leaf_blocking, size_cell_blocking, box)
        {
            m_cell_distrib.resize(tree_height);
        }
        template<typename ParticleContainer>
        explicit dist_group_tree(std::size_t tree_height, std::size_t order, Box const& box,
                                 std::size_t size_leaf_blocking, std::size_t size_cell_blocking,
                                 ParticleContainer const& particle_container,
                                 bool particles_are_sorted = false /*, int in_left_limit = -1*/)
          : base_type(tree_height, order, box, size_leaf_blocking, size_cell_blocking, particle_container,
                      particles_are_sorted /*, in_left_limit*/)
        {
            m_cell_distrib.resize(tree_height);
        }
        template<typename ParticleContainer>
        explicit dist_group_tree(std::size_t tree_height, std::size_t order, Box const& box,
                                 std::size_t size_element_blocking, std::size_t size_cell_blocking)
          : base_type(tree_height, order, box, size_element_blocking, size_cell_blocking)
        {
        }

        void set_leaf_distribution(const data_distrib_type& in_leaf_distrib) { m_leaf_distrib = in_leaf_distrib; }

        void set_cell_distribution(const int in_level, const data_distrib_type& in_cell_distrib)
        {
            m_cell_distrib.at(in_level) = in_cell_distrib;
        }

        void print_distrib(std::ostream& out, bool verbose = true)
        {
            if(m_cell_distrib.size() > 0)
            {
                std::string header;
                if(verbose)
                {
                    out << "Tree distribution" << std::endl;
                }
                for(int l = base_type::top_level(); l < base_type::height(); ++l)
                {
                    if(verbose)
                    {
                        header = "  Level " + std::to_string(l) + " cell distribution: \n";
                    }
                    if(m_cell_distrib[l].size() > 0)
                    {
                        io::print(out, std::move(header), m_cell_distrib[l]);
                    }
                }
                if(verbose)
                {
                    header = "  leaf distribution: \n";
                }
                io::print(out, std::move(header), m_leaf_distrib);
            }
        }

        ~dist_group_tree()
        {
            // std::cout << cpp_tools::colors::red;
            // std::cout << " ~dist_group_tree() " << std::endl;
            // std::cout << cpp_tools::colors::reset;
            // std::cout << " end ~dist_group_tree() " << std::endl;
        }
        template<typename particleContainer>
        void set_particles_in_leaf(particleContainer const& particles)
        {
        }
        template<typename VectorMortonIndexType>
        void create_cells_at_level(const int level, VectorMortonIndexType const& mortonIdx,
                                   VectorMortonIndexType const& ghosts_m2l, data_distrib_value_type const& cell_distrib)
        {
            io::print("create_from_leaf : m2l_ghost", ghosts_m2l);
            io::print(" cell_distrib  ", cell_distrib);
            // construct group of cells at leaf level
            auto first_index = cell_distrib[0];
            auto last =
              std::find_if(ghosts_m2l.begin(), ghosts_m2l.end(), [&first_index](auto& x) { return x > first_index; });
            VectorMortonIndexType ghost_left_mortonIdx(std::distance(ghosts_m2l.begin(), last));
            std::copy(ghosts_m2l.begin(), last, ghost_left_mortonIdx.begin());
            // io::print("create_from_leaf : ghost_left_mortonIdx ", ghost_left_mortonIdx);
            this->build_groups_of_cells_at_level(ghost_left_mortonIdx, level, false);
            this->build_cells_in_groups_at_level(ghost_left_mortonIdx, base_type::m_box, level);

            auto left_block_cells = std::move(base_type::m_group_of_cell_per_level.at(level));
            VectorMortonIndexType ghost_right_mortonIdx(ghosts_m2l.size() - ghost_left_mortonIdx.size());
            std::copy(last, ghosts_m2l.end(), ghost_right_mortonIdx.begin());
            io::print("create_from_leaf : ghost_right_mortonIdx ", ghost_right_mortonIdx);
            this->build_groups_of_cells_at_level(ghost_right_mortonIdx, level, false);
            this->build_cells_in_groups_at_level(ghost_right_mortonIdx, base_type::m_box, level);

            auto right_block_cells = std::move(base_type::m_group_of_cell_per_level.at(level));
            this->build_groups_of_cells_at_level(mortonIdx, level);
            this->build_cells_in_groups_at_level(mortonIdx, base_type::m_box, level);

            auto local_block_cells = std::move(base_type::m_group_of_cell_per_level.at(level));
            auto all_cells_blocks =
              scalfmm::tree::let::merge_blocs(left_block_cells, local_block_cells, right_block_cells);
            std::cout << "  All cells blocks at level " << level << " size: " << all_cells_blocks.size() << std::endl;
            int tt{0};
            for(auto pg: all_cells_blocks)
            {
                std::cout << "block index " << tt++ << " ";
                pg->print();
                std::cout << std::endl;
                // pg->cstorage().print_block_data(std::cout);
            }
            std::cout << std::endl;
            base_type::m_group_of_cell_per_level.at(level) = std::move(all_cells_blocks);
        }
        template<typename VectorLeafInfoType, typename VectorMortonIndexType>
        void create_from_leaf_level(VectorLeafInfoType& localLeaves, VectorLeafInfoType& ghosts_p2p,
                                    VectorMortonIndexType const& ghosts_m2l,
                                    data_distrib_value_type const& leaf_distrib,
                                    data_distrib_value_type const& cell_distrib)
        {
            using morton_type = typename VectorLeafInfoType::value_type::morton_type;
            //
            // compute number of particles
            //
            std::vector<morton_type> mortonIdx;
            std::vector<std::size_t> number_of_part;
            std::tie(mortonIdx, number_of_part) = tree::let::split_structure(localLeaves.cbegin(), localLeaves.cend());
            io::print("create_from_leaf :morton  ", mortonIdx);
            io::print("create_from_leaf :nbpart  ", number_of_part);

            this->build_groups_of_leaves(mortonIdx, number_of_part, base_type::m_box);
            auto localBlocks = std::move(base_type::m_group_of_leaf);

            // Build group on the left
            auto first_index = leaf_distrib[0];
            auto last = std::find_if(ghosts_p2p.begin(), ghosts_p2p.end(),
                                     [&first_index](auto& x) { return x.morton > first_index; });
            std::vector<morton_type> ghost_left_mortonIdx;
            std::vector<std::size_t> ghost_left_number_of_part;
            std::tie(ghost_left_mortonIdx, ghost_left_number_of_part) =
              tree::let::split_structure(ghosts_p2p.begin(), last);
            io::print("create_from_leaf : left morton  ", ghost_left_mortonIdx);
            io::print("create_from_leaf : left nbpart  ", ghost_left_number_of_part);

            this->build_groups_of_leaves(ghost_left_mortonIdx, ghost_left_number_of_part, base_type::m_box, false);
            auto ghost_left_Blocks = std::move(base_type::m_group_of_leaf);
            std::vector<morton_type> ghost_right_mortonIdx;
            std::vector<std::size_t> ghost_right_number_of_part;
            std::tie(ghost_right_mortonIdx, ghost_right_number_of_part) =
              tree::let::split_structure(last, ghosts_p2p.end());

            io::print("create_from_leaf : right morton  ", ghost_right_mortonIdx);
            io::print("create_from_leaf : right nbpart  ", ghost_right_number_of_part);

            this->build_groups_of_leaves(ghost_right_mortonIdx, ghost_right_number_of_part, base_type::m_box, false);
            auto ghost_right_Blocks = std::move(base_type::m_group_of_leaf);

            // Merge the three block structure
            auto all_blocks = scalfmm::tree::let::merge_blocs(ghost_left_Blocks, localBlocks, ghost_right_Blocks);

            base_type::m_group_of_leaf = std::move(all_blocks);
            //  leaves are created
            /////////////////////////////////////////////////////////////////////////////////////
            // same code for the cells we change
            //  - ghosts_p2p in ghosts_m2l
            //  - base_type::m_group_of_leaf in base_type::m_group_of_cell_per_level[leaf_level]
            //
            // we construct the leaves in each group
            io::print("create_from_leaf : m2l_ghost", ghosts_m2l);
            io::print(" cell_distrib  ", cell_distrib);

            auto leaf_level = base_type::m_tree_height - 1;
            this->create_cells_at_level(leaf_level, mortonIdx, ghosts_m2l, cell_distrib);
            //
        }
        /**
         * @brief Set the valid begin and end iterators on cell and leaf group I
         *
         */
        void set_valid_iterators()
        {
            auto& vectG = base_type::m_group_of_leaf;
            for(auto it = std::begin(base_type::m_group_of_leaf); it != std::end(base_type::m_group_of_leaf); ++it)
            {
                if(it->get()->csymbolics().is_mine)
                {
                    base_type::m_view_on_my_leaf_groups[0] = it;
                    break;
                }
            }

            std::cout << " set_valid_iterators: begin " << std::distance(vectG.begin(), base_type::begin_mine_leaves())
                      << std::endl
                      << std::flush;
            for(auto it = std::end(base_type::m_group_of_leaf) - 1; it != std::begin(base_type::m_group_of_leaf); --it)
            {
                if(it->get()->csymbolics().is_mine)
                {
                    auto itpp = it;
                    ++itpp;   // We increment by one to get the final iterator.
                    base_type::m_view_on_my_leaf_groups[1] = itpp;
                    break;
                }
            }
            auto leaf_level = base_type::m_tree_height - 1;
            std::cout << " set_valid_iterators(leaves):\n ";
            std::cout << " level = " << leaf_level << "  begin_mine "
                      << std::distance(vectG.begin(), base_type::begin_mine_leaves()) << " end_mine "
                      << std::distance(vectG.begin(), base_type::end_mine_leaves()) << std::endl;
            ///////////////// End leaves

            std::cout << " set_valid_iterators(cells):\n " << std::flush;

            auto cell_level_it = this->begin_cells() + leaf_level;
            // auto& m_end_my_cells = base_type::m_end_my_cells;
            // auto& m_begin_my_cells = base_type::m_begin_my_cells;

            for(int level = leaf_level; level >= base_type::m_top_level; --level)
            {
                std::cout << "level: " << level << std::endl;
                auto group_of_cell_begin = std::begin(*(cell_level_it));
                auto group_of_cell_end = std::end(*(cell_level_it));
                auto& my_iterator_cells_at_level = base_type::m_view_on_my_cell_groups[level];

                //
                // meta::td<decltype(my_iterator_cells_at_level[0])> u;
                for(auto it = group_of_cell_begin; it != group_of_cell_end; ++it)
                {
                    if(it->get()->csymbolics().is_mine)
                    {
                        my_iterator_cells_at_level[0] = it;
                        break;
                    }
                }
                for(auto it = group_of_cell_end - 1; it != group_of_cell_begin; --it)
                {
                    if(it->get()->csymbolics().is_mine)
                    {
                        my_iterator_cells_at_level[1] = it;
                        ++my_iterator_cells_at_level[1];
                        break;
                    }
                }
                // meta::td<decltype(std::begin(base_type::m_begin_my_cells))> u1;
                // meta::td<decltype(std::begin(base_type::m_view_on_my_cells_grp))> u2;
                std::cout << " level = " << level << "  begin_mine "
                          << std::distance(group_of_cell_begin, base_type::m_view_on_my_cell_groups[level][0])
                          << " end_mine "
                          << std::distance(group_of_cell_begin, base_type::m_view_on_my_cell_groups[level][1])
                          << std::endl;
                --cell_level_it;
            }
        }
        template<typename ParticleContainer>
        auto fill_leaves_with_particles(ParticleContainer const& particle_container) -> void
        {
            //	  using scalfmm::details::tuple_helper;
            // using proxy_type = typename particle_type::proxy_type;
            // using const_proxy_type = typename particle_type::const_proxy_type;
            // using outputs_value_type = typename particle_type::outputs_value_type;
            auto begin_container = std::begin(particle_container);
            std::size_t part_src_index{0};
            std::size_t group_index{0};

            // for(auto pg: m_group_of_leaf)
            // for(auto pg: m_group_of_leaf)
            // for(auto pg = base_type::cbegin_mine_leaves(); pg != base_type::cend_mine_leaves(); ++pg)
            for(auto pg = base_type::cbegin_mine_leaves(); pg != base_type::cend_mine_leaves(); ++pg)
            {
                auto& group = *(pg->get());

                std::size_t leaf_index{0};
                auto leaves_view = group.components();
                // loop on leaves
                for(auto const& leaf: group.components())
                {
                    //     // get the leaf container
                    auto leaf_container_begin = leaf.particles().first;
                    //     // copy the particle in the leaf
                    for(std::size_t index_part = 0; index_part < leaf.size(); ++index_part)
                    {
                        // get the source index in the source container
                        // auto source_index = std::get<1>(tuple_of_indexes.at(part_src_index));
                        // jump to the index in the source container
                        auto jump_to_particle = begin_container;
                        std::advance(jump_to_particle, int(part_src_index));
                        // copy the particle

                        // *leaf_container_begin = particle_container.particle(source_index).as_tuple();
                        // std::cout << part_src_index << " p " << particle_container.at(part_src_index) <<
                        // std::endl;
                        *leaf_container_begin = particle_container.at(part_src_index).as_tuple();

                        //         proxy_type particle_in_leaf(*leaf_container_begin);
                        //         // set the outputs to zero
                        //         for(std::size_t ii{0}; ii < particle_type::outputs_size; ++ii)
                        //         {
                        //             particle_in_leaf.outputs(ii) = outputs_value_type(0.);
                        //         }

                        ++part_src_index;
                        ++leaf_container_begin;
                    }
                    ++leaf_index;
                }
                ++group_index;
                std::cout << " group " << group << std::endl;
            }
#ifdef _DEBUG_BLOCK_DATA
            std::clog << "  FINAL block\n";
            int tt{0};
            for(auto pg: m_group_of_leaf)
            {
                std::clog << "block index " << tt++ << std::endl;
                pg->cstorage().print_block_data(std::clog);
            }
            std::clog << "  ---------------------------------------------------\n";
#endif
        }

      private:
        /// Distribution of leaves at different level. The interval is a range (open on the right)
        data_distrib_type m_leaf_distrib;
        /// Distribution of cells at different level
        std::vector<data_distrib_type> m_cell_distrib;
    };
}   // namespace scalfmm::component

#endif
