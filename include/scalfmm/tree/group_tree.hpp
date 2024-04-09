// --------------------------------
// See LICENCE file at project root
// File : group_tree.hpp
// --------------------------------
#ifndef SCALFMM_TREE_GROUP_TREE_HPP
#define SCALFMM_TREE_GROUP_TREE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "scalfmm/lists/policies.hpp"
#include "scalfmm/lists/sequential.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/tools/bench.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/group.hpp"
#ifdef _OMP
#include "scalfmm/lists/omp.hpp"
#endif
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/sort.hpp"

#error("Do not use this file)
namespace scalfmm::component
{
    template<typename Cell, typename Leaf, typename Box = box<typename Leaf::position_type>>
    class group_tree
    {
      public:
        using cell_type = Cell;
        using leaf_type = Leaf;
        using particle_type = typename Leaf::particle_type;
        using group_of_cell_type = group<cell_type>;
        using group_of_leaf_type = group<leaf_type>;
        using box_type = Box;
        using position_type = typename box_type::position_type;
        using position_value_type = typename box_type::value_type;
        static constexpr std::size_t dimension = box_type::dimension;
        using cell_group_vector_type = std::vector<std::vector<std::shared_ptr<group_of_cell_type>>>;
        using leaf_group_vector_type = std::vector<std::shared_ptr<group_of_leaf_type>>;

        // iterators
        using cell_iterator_type = typename cell_group_vector_type::iterator;
        using const_cell_iterator_type = typename cell_group_vector_type::const_iterator;
        using leaf_iterator_type = typename leaf_group_vector_type::iterator;
        using const_leaf_iterator_type = typename leaf_group_vector_type::const_iterator;

        using iterator_type = std::tuple<leaf_iterator_type, cell_iterator_type>;
        using const_iterator_type = std::tuple<const_leaf_iterator_type, const_cell_iterator_type>;

        group_tree() = default;
        group_tree(group_tree const&) = default;
        group_tree(group_tree&&) noexcept = default;
        inline auto operator=(group_tree const&) -> group_tree& = default;
        inline auto operator=(group_tree&&) noexcept -> group_tree& = default;
        ~group_tree() = default;

      private:
        const std::size_t m_tree_height{};
        int m_tree_levels_above_root{};    ///< the number of level above the root needed in periodic simulation
        const std::size_t m_top_level{};   ///< the level to stop the FMM algorithm (generally 2)
        const std::size_t m_order{};
        const std::size_t m_number_of_leaves_per_group{};
        const std::size_t m_number_of_component_per_group{};
        const std::size_t m_number_of_cells_per_group{};
        bool m_interaction_p2p_lists_built{false};
        bool m_interaction_m2l_lists_built{false};

        cell_group_vector_type m_group_of_cell_per_level;
        leaf_group_vector_type m_group_of_leaf;
        box_type m_box{};

      public:
        [[nodiscard]] auto box_center() const { return m_box.center(); }
        [[nodiscard]] inline auto leaf_level() const noexcept -> std::size_t { return m_tree_height - 1; }
        [[nodiscard]] inline auto top_level() const noexcept -> std::size_t { return m_top_level; }

        [[nodiscard]] auto box_width(std::size_t dimension = 0) const { return m_box.width(dimension); }
        [[nodiscard]] auto leaf_width(std::size_t dimension = 0) const
        {
            return m_box.width(dimension) / (1 << leaf_level());
        }
        [[nodiscard]] auto& is_interaction_m2l_lists_built() { return m_interaction_m2l_lists_built; }
        [[nodiscard]] auto& is_interaction_p2p_lists_built() { return m_interaction_p2p_lists_built; }

      private:
        /// @brief This function takes a vector of tuple storing indices.
        /// In the the tuple the first element is the morton index of the particle.
        /// The second is the index of the particle in the particle container.
        /// The function will count the duplicated morton index i.e the number of
        /// particles to store in the leaf of the corresponding morton index.
        /// ex : if in the vector the morton index 2 appears 4 times
        /// -> 4 particles in the leaf of morton index 2.
        /// Then the function removes the duplicates of morton indices.
        /// It returns the vector holding the number of particle per leaves.
        //
        /// @tparam MortonType : the type of the morton index stored in the vector
        /// @param vector_of_mortons : a vector holding indices for permutation of size number_of_particles
        /// and resized to the number_of_leaves when the function returns.
        ///
        /// @return : a vector of std::size_t of size number_of_leaves.
        template<typename MortonType>
        auto get_leaves_distribution(std::vector<MortonType>& vector_of_mortons) -> std::vector<std::size_t>
        {
            // vector to store the number of particles per leaves
            // here we count the duplicates of morton indices to know the number of particles

            std::vector<std::size_t> number_of_particles_per_leaves{};

            std::size_t i{0};
            for(auto it = std::cbegin(vector_of_mortons); it != std::cend(vector_of_mortons);)
            {
                auto dups{0};
                auto target{*it};
                while(*it == target && it != std::cend(vector_of_mortons))
                {
                    ++it;
                    ++dups;
                }
                number_of_particles_per_leaves.push_back(dups);
            }
            // here, we remove the duplicates of the morton indices. It will give us the morton indexes of the leaves.
            auto last = std::unique(std::begin(vector_of_mortons), std::end(vector_of_mortons));
            // we erase the last part of the vector (the residual space of duplicates) as it is underterminate.
            vector_of_mortons.erase(last, std::end(vector_of_mortons));

            assertm(vector_of_mortons.size() == number_of_particles_per_leaves.size(), "wrong number of leaves !");

            stats(vector_of_mortons, number_of_particles_per_leaves);

            return number_of_particles_per_leaves;
        }

        /// @brief It builds the groups of leaves for the leaf level.
        ///
        /// @tparam MortonType : the type of the morton index stored in the vector
        /// @param vector_of_mortons : a vector holding indices for permutation of size number_of_leaves
        /// @param number_of_particles_per_leaves: a vector holding the number of particles per leaves.
        ///
        /// @return : void
        template<typename MortonType>
        auto build_groups_of_leaves(std::vector<MortonType> const& vector_of_mortons) -> void
        {
            auto number_of_leaves{vector_of_mortons.size()};
            // the number of groups at leaf level
            auto number_of_groups_at_leaf_level{number_of_leaves / m_number_of_leaves_per_group};
            // the rest of the leaves to store in the last group of a smaller size.
            auto remain_number_of_leaves{number_of_leaves % m_number_of_leaves_per_group};

            // resize the vector of groups of leaves
            m_group_of_leaf.resize(number_of_groups_at_leaf_level);

            // loop on the full groups
            for(std::size_t g{0}; g < number_of_groups_at_leaf_level; ++g)
            {
                // start and end of the leaves
                auto start_index{g * m_number_of_leaves_per_group};
                auto end_index{(g * m_number_of_leaves_per_group) + m_number_of_leaves_per_group - 1};

                // get the starting and ending morton indices i.e the first and last leaves morton indices
                auto starting_morton_index{vector_of_mortons.at(start_index)};
                auto ending_morton_index{vector_of_mortons.at(end_index)};
                // we create the group and stores the pointer in the vector of groups
                m_group_of_leaf.at(g) = std::move(std::make_shared<group_of_leaf_type>(
                  starting_morton_index, ending_morton_index + 1, m_number_of_leaves_per_group, g, true));
            }

            // here is the residue of the leaves
            if(remain_number_of_leaves != 0)
            {
                // we go at the end of the full groups
                auto start_index{number_of_groups_at_leaf_level * m_number_of_leaves_per_group};
                auto end_index{(number_of_groups_at_leaf_level * m_number_of_leaves_per_group) +
                               remain_number_of_leaves - 1};

                // we get the morton indices
                auto starting_morton_index{vector_of_mortons.at(start_index)};
                auto ending_morton_index{vector_of_mortons.at(end_index)};

                // we create the last group and store the pointer at the end of the vector of groups
                m_group_of_leaf.push_back(std::move(
                  std::make_shared<group_of_leaf_type>(starting_morton_index, ending_morton_index + 1,
                                                       remain_number_of_leaves, number_of_groups_at_leaf_level, true)));
            }
        }

        /// @brief This function builds the groups of cells at the specified level
        ///
        /// @tparam MortonIndex : the type of the morton index
        /// @param vector_of_mortons : the vector holding the morton indices of the cells at the specified level
        /// @param level : the level to build the group
        ///
        /// @return void
        template<typename MortonIndex>
        auto build_groups_of_cells_at_level(std::vector<MortonIndex> const& vector_of_mortons, std::size_t level)
          -> void
        {
            auto number_of_cells{vector_of_mortons.size()};
            auto number_of_groups{number_of_cells / m_number_of_cells_per_group};
            // the rest of the cells to store in the last group of a smaller size.
            auto remain_number_of_cells{number_of_cells % m_number_of_cells_per_group};

            // resizing of the level
            m_group_of_cell_per_level.at(level).resize(number_of_groups);

            for(std::size_t g{0}; g < number_of_groups; ++g)
            {
                // start and end of the cells
                auto start_index{g * m_number_of_cells_per_group};
                auto end_index{(g * m_number_of_cells_per_group) + m_number_of_cells_per_group - 1};
                // get the starting and ending morton indices i.e the first and last cells morton indices
                auto starting_morton_index{vector_of_mortons.at(start_index)};
                auto ending_morton_index{vector_of_mortons.at(end_index)};
                // Create a group
                m_group_of_cell_per_level.at(level).at(g) = std::move(std::make_shared<group_of_cell_type>(
                  starting_morton_index, ending_morton_index + 1, m_number_of_cells_per_group, g, true));
            }

            if(remain_number_of_cells != 0)
            {
                // we go at the end of the full groups
                auto start_index{number_of_groups * m_number_of_cells_per_group};
                auto end_index{(number_of_groups * m_number_of_cells_per_group) + remain_number_of_cells - 1};

                // we get the morton indices
                auto starting_morton_index{vector_of_mortons.at(start_index)};
                auto ending_morton_index{vector_of_mortons.at(end_index)};
                m_group_of_cell_per_level.at(level).push_back(std::move(std::make_shared<group_of_cell_type>(
                  starting_morton_index, ending_morton_index + 1, remain_number_of_cells, number_of_groups, true)));
            }
        }

        /// @brief This function constructs the leaves in each group.
        ///
        /// @tparam MortonType : the type of the morton index stored in the vector
        /// @param vector_of_mortons : a vector holding indices for permutation of size number_of_leaves
        /// @param number_of_particles_per_leaves : a vector holding the number of particles per leaves.
        /// @param box : the simulation box
        ///
        /// @return void
        template<typename MortonType>
        auto build_leaves_in_groups(std::vector<MortonType> const& vector_of_mortons,
                                    std::vector<std::size_t> const& number_of_particles_per_leaves, box_type const& box)
          -> void
        {
            std::size_t leaf_index{0};
            // loop on groups
            for(std::size_t ng{0}; ng < m_group_of_leaf.size(); ++ng)
            {
                auto pg = m_group_of_leaf.at(ng);
                auto& group_symbolics{pg->symbolics()};
                // loop on leaves
                for(auto&& leaf: *pg)
                {
                    auto morton_index_of_leaf = vector_of_mortons.at(leaf_index);
                    // get the coordinate of the leaf in the tree
                    auto coordinate = index::get_coordinate_from_morton_index<dimension>(morton_index_of_leaf);
                    // get the corresponding box to the leaf
                    auto width_center{
                      index::get_box(box.c1(), box.width(0), coordinate, m_tree_height, m_tree_height - 1)};

                    leaf = std::move(leaf_type(number_of_particles_per_leaves.at(leaf_index), std::get<1>(width_center),
                                               std::get<0>(width_center), morton_index_of_leaf));
                    // accumulate for set the number of particle in group.
                    group_symbolics.number_of_particles_in_group += number_of_particles_per_leaves.at(leaf_index);
                    ++leaf_index;
                }
            }
        }

        /// @brief This function builds the cells in the groups at the specified level
        ///
        /// @tparam MortonType : the type of the morton index
        /// @param vector_of_mortons : a vector holding the morton indices of the cells at the specified level
        /// @param box : the box simulation
        /// @param level : the level to construct the groups
        ///
        /// @return void
        template<typename MortonType>
        auto build_cells_in_groups_at_level(std::vector<MortonType> const& vector_of_mortons, box_type const& box,
                                            std::size_t level) -> void
        {
            std::size_t cell_index{0};
            // loop on groups
            for(std::size_t ng{0}; ng < m_group_of_cell_per_level.at(level).size(); ++ng)
            {
                auto pg = m_group_of_cell_per_level.at(level).at(ng);
                // loop on leaves
                for(auto&& cell: *pg)
                {
                    auto morton_index_of_cell = vector_of_mortons.at(cell_index);
                    // get the coordinate of the leaf in the tree
                    auto coordinate = index::get_coordinate_from_morton_index<dimension>(morton_index_of_cell);
                    // get the corresponding box to the leaf
                    auto width_center{index::get_box(box.c1(), box.width(0), coordinate, m_tree_height, level)};

                    cell = std::move(cell_type(std::get<1>(width_center), std::get<0>(width_center), m_order, level,
                                               morton_index_of_cell, coordinate));
                    ++cell_index;
                }
            }
        }

      public:
        /// @brief Constructor of the group tree.
        /// It initialized all levels with leaves and cells from the particle container passed.
        ///
        /// @tparam ParticleContainer : the type of the particle container
        /// @param tree_height : the height of the tree
        /// @param order : order of the simulation
        /// @param box : the box of the simulation
        /// @param number_of_leaves_per_group : blocking on the leaves
        /// @param number_of_cells_per_group : blocking on the cells
        /// @param particle_container : the container holding the particles
        /// @param particles_are_sorted : true if the particles are sorted, false either
        /// @param in_top_level : last level of cells
        template<typename ParticleContainer>
        group_tree(std::size_t tree_height, std::size_t order, box_type const& box,
                   std::size_t number_of_leaves_per_group, std::size_t number_of_cells_per_group,
                   ParticleContainer const& particle_container, bool particles_are_sorted = false)
          : m_tree_height(tree_height)
          , m_tree_levels_above_root(-1)
          , m_top_level((box.is_periodic() ? 0 : 2))
          , m_order(order)
          , m_number_of_leaves_per_group(number_of_leaves_per_group)
          , m_number_of_cells_per_group(number_of_cells_per_group)
          , m_group_of_cell_per_level(tree_height)
          , m_interaction_p2p_lists_built(false)
          , m_interaction_m2l_lists_built(false)
          , m_box(box)
        {
            static_assert(
              std::is_same_v<typename ParticleContainer::value_type, particle_type>,
              "group_tree : Particles contain in leafs are not the same as the ones in the container passed "
              "as argument to this constructor.");
            auto number_of_particles{particle_container.size()};
            const std::size_t leaf_level{m_tree_height - 1};

            // First we work at leaf level
            // Convert position to morton index

            /// Build the permutation to sort the particles according to their
            /// morton. tuple_of_indexes is a vector of a tuple (idx, morton
            /// idx)
            auto tuple_of_indexes =
              scalfmm::utils::get_morton_permutation(m_box, leaf_level, particle_container, particles_are_sorted);

            // extract the vector of morton index
            std::vector<std::size_t> vector_of_mortons(tuple_of_indexes.size());
            std::transform(std::begin(tuple_of_indexes), std::end(tuple_of_indexes), std::begin(vector_of_mortons),
                           [](auto const& t) { return std::get<0>(t); });

            // construct all levels
            this->construct(vector_of_mortons);
            // then, we fill each leaf with its particle
            this->fill_leaves_with_particles(tuple_of_indexes, particle_container);
        }

        template<typename MortonType, typename = typename std::enable_if_t<std::is_integral_v<MortonType>>>
        group_tree(std::size_t tree_height, std::size_t order, box_type const& box,
                   std::size_t number_of_leaves_per_group, std::size_t number_of_cells_per_group,
                   std::vector<MortonType> const& vector_of_mortons)
          : m_tree_height(tree_height)
          , m_tree_levels_above_root(-1)
          , m_top_level((box.is_periodic() ? 0 : 2))
          , m_order(order)
          , m_number_of_leaves_per_group(number_of_leaves_per_group)
          , m_number_of_cells_per_group(number_of_cells_per_group)
          , m_group_of_cell_per_level(tree_height)
          , m_interaction_p2p_lists_built(false)
          , m_interaction_m2l_lists_built(false)
          , m_box(box)
        {
            this->construct(vector_of_mortons);
        }

        template<typename ParticleContainer>
        group_tree(std::size_t tree_height, std::size_t order, box_type const& box,
                   std::size_t number_of_component_per_group, ParticleContainer const& particle_container,
                   bool particles_are_sorted = false)
        {
            group_tree(tree_height, order, box, number_of_component_per_group, number_of_component_per_group,
                       particle_container, particles_are_sorted);
        }

        template<typename MortonType, typename = typename std::enable_if_t<std::is_integral_v<MortonType>>>
        group_tree(std::size_t tree_height, std::size_t order, box_type const& box,
                   std::size_t number_of_component_per_group, std::vector<MortonType> const& vector_of_mortons)
        {
            group_tree(tree_height, order, box, number_of_component_per_group, number_of_component_per_group,
                       vector_of_mortons);
        }

        template<typename MortonType>
        inline auto construct(std::vector<MortonType>& vector_of_mortons) -> void
        {
            // vector to store the number of particles per leaves
            auto number_of_particles_per_leaves{get_leaves_distribution(vector_of_mortons)};
            // we build the first level of group for the leaves
            this->build_groups_of_leaves(vector_of_mortons);

            // we construct the leaves in each group
            this->build_leaves_in_groups(vector_of_mortons, number_of_particles_per_leaves, m_box);
            auto leaf_level = m_tree_height - 1;
            // construct group of cells at leaf level

            this->build_groups_of_cells_at_level(vector_of_mortons, leaf_level);
            // construct cells in group of leaf level

            this->build_cells_in_groups_at_level(vector_of_mortons, m_box, leaf_level);
            // loop on levels  leaf_level -1  = first level of cells
            auto top_level = m_box.is_periodic() ? 0 : 2;
            auto down_level = leaf_level - 1;

            for(int level{int(down_level)}; level >= int(top_level); --level)
            {
                // update vector_of_mortons for upper level
                index::get_parent_morton_indices(vector_of_mortons, dimension);
                // construct group of cells at current level
                build_groups_of_cells_at_level(vector_of_mortons, level);
                // construct cells in group of current level
                build_cells_in_groups_at_level(vector_of_mortons, m_box, level);
            }
        }

        template<typename MortonType>
        inline auto construct(std::vector<MortonType>& vector_of_mortons, box_type const& box) -> void
        {
            m_box = box;
            this->construct(vector_of_mortons);
        }

        /// @brief The function fills the particle in each leaf.
        /// Its uses the source index in the tuple of indices to get the
        /// particle from the source container
        ///
        /// @tparam VectorOfTuples : a vector of tuples of indices of size number_of_leaves
        /// @tparam ParticleContainer : the particle container
        /// @param tuple_of_indexes  : a vector holding indices for permutation of size number_of_leaves
        /// @param particle_container : the container storing all the particles.
        ///
        /// @return
        template<typename VectorOfTuples, typename ParticleContainer>
        auto fill_leaves_with_particles(VectorOfTuples const& tuple_of_indexes,
                                        ParticleContainer const& particle_container) -> void
        {
            using proxy_type = typename particle_type::proxy_type;
            using outputs_value_type = typename particle_type::outputs_value_type;

            auto begin_container = std::begin(particle_container);
            std::size_t leaf_index{0};
            // loop on groups
            for(auto pg: m_group_of_leaf)
            {
                // loop on leaves
                for(auto& leaf: *pg)
                {
                    // get the leaf container
                    auto leaf_container_begin = std::begin(leaf.particles());
                    // copy the particle in the leaf
                    for(std::size_t index_part = 0; index_part < leaf.size(); ++index_part)
                    {
                        // get the source index in the source container
                        auto source_index = std::get<1>(tuple_of_indexes.at(leaf_index++));
                        // jump to the index in the source container
                        auto jump_to_particle = begin_container;
                        std::advance(jump_to_particle, source_index);
                        // copy the particle
                        *leaf_container_begin = meta::as_tuple(*jump_to_particle);
                        proxy_type particle_in_leaf(*leaf_container_begin);
                        // set the outputs to zero
                        for(std::size_t ii{0}; ii < particle_type::outputs_size; ++ii)
                        {
                            particle_in_leaf.outputs(ii) = outputs_value_type(0.);
                        }
                        ++leaf_container_begin;
                    }
                }
            }
        }

        /**
         * @brief Construct the P2P and M2L interaction lists
         *
         * @param[in] tree_source the tree containing the source cells and leaves
         * @param[in] neighbour_separation separation criterion use to separate teh near and the far field
         * @param[in] mutual boolean to specify if the direct pass use a symmetric algorithm (mutual interactions)
         * @param[in] policy the policy to compute the interaction list (sequential, omp )
         */
        template<typename TREE_EXT>
        inline auto build_interaction_lists(TREE_EXT const& tree_source, const int& neighbour_separation,
                                            const bool mutual, const int& policy = scalfmm::list::policies::sequential)
          -> void
        {
            switch(policy)
            {
            case list::policies::sequential:
                scalfmm::list::sequential::build_interaction_lists(tree_source, *this, neighbour_separation, mutual);
                break;
#ifdef _OMP
            case list::policies::omp:
                scalfmm::list::omp::build_interaction_lists(tree_source, *this, neighbour_separation, mutual);
                break;
#endif
            default:
                scalfmm::list::sequential::build_interaction_lists(tree_source, *this, neighbour_separation, mutual);
            }
        }

        ///
        /// \brief  insert cells associated to Morton index at level in the tree
        ///
        ///
        /// \param level level in the tree to add component associated to Morton
        /// \param cells_index the indexes of the cells to add
        ///
        ///
        template<typename VectorMortonIdx>
        inline auto insert_cells_at_level(const int& level, const VectorMortonIdx& cells_index)
        {
            std::cerr << " insert_cells_at_level not yet implemented" << std::endl;
            throw std::runtime_error("insert_cells_at_level not yet implemented");
        }
        ///
        /// \brief  insert leaves associated to Morton index at level in the tree
        ///
        ///
        /// \param leaves_index the indexes of the leaves to add
        /// indes
        ///
        template<typename VectorMortonIdx>
        inline auto insert_leaves(const VectorMortonIdx& leaves_index)
        {
            std::cerr << " insert_leaves not yet implemented" << std::endl;
            throw std::runtime_error("insert_leaves not yet implemented");
        }

        // /// @brief reset all
        // ///
        // /// @return
        // inline auto reset_particles()
        // {
        //     for(auto pg: m_group_of_leaf)
        //     {
        //         // loop on leaves
        //         for(auto& leaf: *pg)
        //         {
        //             leaf.particles().clear();
        //         }
        //     }
        // }

        template<typename MortonType>
        auto stats(std::vector<MortonType> const& vector_of_mortons,
                   std::vector<std::size_t> const& number_of_particles_per_leaves) -> void
        {
            auto min =
              std::min_element(std::begin(number_of_particles_per_leaves), std::end(number_of_particles_per_leaves));
            auto max =
              std::max_element(std::begin(number_of_particles_per_leaves), std::end(number_of_particles_per_leaves));
            auto sum =
              std::accumulate(std::begin(number_of_particles_per_leaves), std::end(number_of_particles_per_leaves), 0.);

            auto mean = sum / number_of_particles_per_leaves.size();

            double sq_sum =
              std::inner_product(std::begin(number_of_particles_per_leaves), std::end(number_of_particles_per_leaves),
                                 std::begin(number_of_particles_per_leaves), 0.0);

            double stdev = std::sqrt(sq_sum / number_of_particles_per_leaves.size() - mean * mean);

            std::cout << "[stats][min:max] : " << *min << ':' << *max << "\n";
            std::cout << "[stats][mean]    : " << mean << "\n";
            std::cout << "[stats][stddev]  : " << stdev << "\n";

            // for(std::size_t i{0}; i < vector_of_mortons.size(); ++i)
            //{
            //   bench::dump_csv( "distleaves.csv"
            //                  , "Morton,particles"
            //                  , std::to_string(vector_of_mortons[i])
            //                  , std::to_string(number_of_particles_per_leaves[i])
            //                  );
            // }
        }

        ///
        /// \brief trace the index of the cells and leaves in the tree
        ///
        /// Depending on the level we print more or less details
        ///  level_trace = 1 print minimal information (height, order, group size)
        ///  level_trace = 2 print information of the tree (group interval and index inside)
        ///  level_trace = 3 print information of the tree (leaf interval and index inside and their p2p interaction
        ///  list) level_trace = 4 print information of the tree (cell interval and index inside and their m2l
        ///  interaction list)
        /// level_trace = 5 print information of the tree (leaf and cell interval and index inside
        ///  and their p2p and m2l interaction lists)
        ///
        /// @warning to have the right p2p list we have to have the group size in the tree equal to the number
        ///  of leaves otherwise, we only print the index inside the group.
        ///
        /// @param[in] level_trace level of the trace
        ///
        inline auto trace(std::ostream& os, const std::size_t level_trace = 0) -> void
        {
            scalfmm::utils::trace(os, *this,level_trace);
        }

        [[nodiscard]] inline auto height() const noexcept -> std::size_t { return m_tree_height; }

        [[nodiscard]] inline auto levels_above_root() const noexcept -> std::size_t { return m_tree_levels_above_root; }

        inline auto set_levels_above_root(const int in_nb_levels) noexcept -> void
        {
            m_tree_levels_above_root = in_nb_levels;
        }

        [[nodiscard]] inline auto box() const noexcept -> box_type const& { return m_box; }

        [[nodiscard]] inline auto order() const noexcept -> std::size_t { return m_order; }
        
        [[nodiscard]] inline auto group_of_leaves() const noexcept -> leaf_group_vector_type
        {
            return m_group_of_leaf;
        }
        // /**
        //  * @brief return the vector of shared pointer of groups
        //  *
        //  * @return leaf_group_vector_type
        //  */
        // [[nodiscard]] inline auto vector_of_groups() const noexcept -> leaf_group_vector_type& { return
        // m_group_of_leaf; }

        [[nodiscard]] inline auto group_of_leaf_size() const noexcept -> std::size_t
        {
            return m_number_of_leaves_per_group;
        }
        [[nodiscard]] inline auto leaf_group_size() const noexcept -> std::size_t { return m_group_of_leaf.size(); }

        [[nodiscard]] inline auto group_of_cell_size() const noexcept -> std::size_t
        {
            return m_number_of_cells_per_group;
        }

        // iterators related
        /**
         * @brief return the begin iterator on the groups for each level
         *
         *
         * @return tuple of begin iterator organized as follows (leaf, leaf_cell, ... root_cell)
         */
        [[nodiscard]] inline auto begin() -> iterator_type
        {
            return std::make_tuple(std::begin(m_group_of_leaf), std::begin(m_group_of_cell_per_level));
        }

        [[nodiscard]] inline auto begin() const -> const_iterator_type
        {
            return std::make_tuple(std::cbegin(m_group_of_leaf), std::cbegin(m_group_of_cell_per_level));
        }

        [[nodiscard]] inline auto cbegin() const -> const_iterator_type
        {
            return std::make_tuple(std::cbegin(m_group_of_leaf), std::cbegin(m_group_of_cell_per_level));
        }
        /**
         * @brief return the end iterator on the groups for each level
         *
         *
         * @return tuple of end iterator organized as follows (leaf, leaf_cell, ... root_cell)
         */
        [[nodiscard]] inline auto end() -> iterator_type
        {
            return std::make_tuple(std::end(m_group_of_leaf), std::end(m_group_of_cell_per_level));
        }

        [[nodiscard]] inline auto end() const -> const_iterator_type
        {
            return std::make_tuple(std::cend(m_group_of_leaf), std::cend(m_group_of_cell_per_level));
        }

        [[nodiscard]] inline auto cend() const -> const_iterator_type
        {
            return std::make_tuple(std::cend(m_group_of_leaf), std::cend(m_group_of_cell_per_level));
        }

        [[nodiscard]] inline auto begin_leaves() -> leaf_iterator_type { return std::begin(m_group_of_leaf); }

        [[nodiscard]] inline auto begin_leaves() const -> const_leaf_iterator_type
        {
            return std::cbegin(m_group_of_leaf);
        }

        [[nodiscard]] inline auto cbegin_leaves() const -> const_leaf_iterator_type
        {
            return std::cbegin(m_group_of_leaf);
        }

        [[nodiscard]] inline auto begin_cells() -> cell_iterator_type { return std::begin(m_group_of_cell_per_level); }

        [[nodiscard]] inline auto begin_cells() const -> const_cell_iterator_type
        {
            return std::cbegin(m_group_of_cell_per_level);
        }

        [[nodiscard]] inline auto cbegin_cells() const -> const_cell_iterator_type
        {
            return std::cbegin(m_group_of_cell_per_level);
        }

        [[nodiscard]] inline auto end_leaves() -> leaf_iterator_type { return std::end(m_group_of_leaf); }

        [[nodiscard]] inline auto end_leaves() const -> const_leaf_iterator_type { return std::cend(m_group_of_leaf); }

        [[nodiscard]] inline auto cend_leaves() const -> const_leaf_iterator_type { return std::cend(m_group_of_leaf); }

        [[nodiscard]] inline auto end_cells() -> cell_iterator_type { return std::end(m_group_of_cell_per_level); }

        [[nodiscard]] inline auto end_cells() const -> const_cell_iterator_type
        {
            return std::cend(m_group_of_cell_per_level);
        }

        [[nodiscard]] inline auto cend_cells() const -> const_cell_iterator_type
        {
            return std::cend(m_group_of_cell_per_level);
        }

        inline void reset_outputs()
        {
            // leaf level update only P2P ???
            component::for_each(std::get<0>(begin()), std::get<0>(end()),
                                [this](auto& group)
                                {
                                    std::size_t index_in_group{0};
                                    component::for_each(std::begin(*group), std::end(*group),
                                                        [&group, &index_in_group, this](auto& leaf)
                                                        { leaf.reset_outputs(); });
                                });
        }

        inline void reset_far_field()
        {
            auto cell_level_it = this->cbegin_cells() + (m_tree_height - 1);

            int top_level = m_box.is_periodic() ? 0 : 2;
            for(int level = int(m_tree_height) - 1; level >= top_level; --level)
            {
                auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                auto group_of_cell_end = std::cend(*(cell_level_it));
                std::for_each(group_of_cell_begin, group_of_cell_end,
                              [](auto const& ptr_group)
                              {
                                  auto const& current_group_symbolics = ptr_group->csymbolics();
                                  component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                      [](auto& cell)
                                                      {
                                                          cell.reset_multipoles();
                                                          cell.reset_locals();
                                                      });
                              });
                --cell_level_it;
            }
        }
        inline void reset_multipoles()
        {
            auto cell_level_it = this->cbegin_cells() + (m_tree_height - 1);

            int top_level = m_box.is_periodic() ? 0 : 2;
            for(int level = int(m_tree_height) - 1; level >= top_level; --level)
            {
                auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                auto group_of_cell_end = std::cend(*(cell_level_it));
                std::for_each(group_of_cell_begin, group_of_cell_end,
                              [](auto const& ptr_group)
                              {
                                  auto const& current_group_symbolics = ptr_group->csymbolics();
                                  component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                      [](auto& cell) { cell.reset_multipoles(); });
                              });
                --cell_level_it;
            }
        }
        inline void reset_locals()
        {
            auto cell_level_it = this->cbegin_cells() + (m_tree_height - 1);

            int top_level = m_box.is_periodic() ? 0 : 2;
            for(int level = int(m_tree_height) - 1; level >= top_level; --level)
            {
                auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                auto group_of_cell_end = std::cend(*(cell_level_it));
                std::for_each(group_of_cell_begin, group_of_cell_end,
                              [](auto const& ptr_group)
                              {
                                  auto const& current_group_symbolics = ptr_group->csymbolics();
                                  component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                      [](auto& cell) { cell.reset_locals(); });
                              });
                --cell_level_it;
            }
        }
    };

}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_GROUP_TREE_HPP
