// --------------------------------
// See LICENCE file at project root
// File : group_of_particles.hpp
// --------------------------------
#ifndef SCALFMM_TREE_GROUP_OF_PARTICLES_HPP
#define SCALFMM_TREE_GROUP_OF_PARTICLES_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <vector>

#include "scalfmm/container/block.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/tree/header.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/massert.hpp"

namespace scalfmm::component
{
    /**
     * @brief Group class class manages leaves in block allocation
     *
     * The group is composed of two elements. The storage which contains the
     *   particles or multipole/local values and symbolic information of the
     *   components in the group and an array of components (view) that allows
     *   to access the data inside the storage.
     */
    template<typename Leaf, typename Particle>
    struct group_of_particles
    {
      public:
        using particle_type = Particle;
        using component_type = Leaf;
        using symbolics_type = symbolics_data<group_of_particles<component_type, particle_type>>;
        using symbolics_component_type = symbolics_data<component_type>;
        using block_type = std::vector<component_type>;
        using iterator_type = typename block_type::iterator;
        using const_iterator_type = typename block_type::const_iterator;
        using storage_type = typename symbolics_type::storage_type;

        group_of_particles() = default;
        group_of_particles(group_of_particles const&) = default;
        group_of_particles(group_of_particles&&) noexcept = default;
        inline auto operator=(group_of_particles const&) -> group_of_particles& = default;
        inline auto operator=(group_of_particles&&) noexcept -> group_of_particles& = default;
        ~group_of_particles() = default;

        /**
         * @brief Construct a new group of particles object
         *
         * @param starting_morton_idx  Morton index of the first object (leaf or cell) inside the group
         * @param ending_morton_idx    Morton index of the last object (leaf or cell) inside the group
         * @param number_of_component  Number of leaves or cells inside the group
         * @param number_of_particles_in_group  Number of particles inside the group
         * @param index_global         Global index of the group
         * @param is_mine              I am the owner of the group
         */
        group_of_particles(std::size_t starting_morton_idx, std::size_t ending_morton_idx,
                           std::size_t number_of_component, std::size_t number_of_particles_in_group,
                           std::size_t index_global, bool is_mine)
          : m_vector_of_components(number_of_component)
          , m_components_storage(number_of_particles_in_group, number_of_component)
          , m_number_of_component{number_of_component}
        {
            auto& group_symbolics = m_components_storage.header();

            group_symbolics.starting_index = starting_morton_idx;
            group_symbolics.ending_index = ending_morton_idx;
            group_symbolics.number_of_component_in_group = number_of_component;
            group_symbolics.number_of_particles_in_group = number_of_particles_in_group;
            group_symbolics.idx_global = index_global;
            group_symbolics.is_mine = is_mine;
        }
        // /**
        //  * @brief Rebuilt all the pointers inside the vector of blocks
        //  *
        //  */
        // auto rebuilt_leaf_view() -> void
        // {
        //     std::cout << "rebuilt_leaf_view nb view " << m_number_of_component << "   " <<
        //     m_vector_of_components.size()
        //               << std::endl;
        // }
        /**
         * @brief  Display the elements of the group (views and the storage)
         *
         * @param os
         * @param group
         * @return std::ostream&
         */
        inline friend auto operator<<(std::ostream& os, const group_of_particles& group) -> std::ostream&
        {
             os << cpp_tools::colors::green;
             os << "group:" << group.csymbolics().idx_global << " nb_leaves: " << group.size() << " is_mine "
                << group.csymbolics().is_mine << std::endl;
             os << "leaves_view: \n";
             for(auto v: group.components())
             {
                 os << v << "\n";
             }
            os << "storage: \n";
            os << group.cstorage() << std::endl;
            os  << cpp_tools::colors::reset;
            return os;
        }
        /**
         * @brief return the symbolic structure of the group_of_particles
         *
         * @return symbolics_type&
         */
        [[nodiscard]] inline auto symbolics() -> symbolics_type& { return m_components_storage.header(); }
        [[nodiscard]] inline auto symbolics() const -> symbolics_type const& { return m_components_storage.header(); }
        [[nodiscard]] inline auto csymbolics() const -> symbolics_type const& { return m_components_storage.cheader(); }
        /**
         * @brief  Get the vector of components (leaves)
         *
         * @return block_type&
         */
        [[nodiscard]] inline auto components() -> block_type& { return m_vector_of_components; }
        [[nodiscard]] inline auto components() const -> block_type const& { return m_vector_of_components; }
        [[nodiscard]] inline auto ccomponents() const -> block_type const& { return m_vector_of_components; }
        /**
         * @brief return the storage of the particles inside the group
         *
         * @return storage_type&
         */
        [[nodiscard]] inline auto storage() -> storage_type& { return m_components_storage; }
        [[nodiscard]] inline auto storage() const -> storage_type const& { return m_components_storage; }
        [[nodiscard]] inline auto cstorage() const -> storage_type const& { return m_components_storage; }
        /**
         * @brief get the first component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto begin() -> iterator_type { return std::begin(m_vector_of_components); }
        [[nodiscard]] inline auto begin() const -> const_iterator_type { return std::cbegin(m_vector_of_components); }
        [[nodiscard]] inline auto cbegin() const -> const_iterator_type { return std::cbegin(m_vector_of_components); }
        /**
         * @brief get the last component iterator
         *
         * @return iterator_type
         */
        [[nodiscard]] inline auto end() -> iterator_type { return std::end(m_vector_of_components); }
        [[nodiscard]] inline auto end() const -> const_iterator_type { return std::cend(m_vector_of_components); }
        [[nodiscard]] inline auto cend() const -> const_iterator_type { return std::cend(m_vector_of_components); }
        /**
         * @brief Return the number of components (cells or leaves)
         *
         */
        [[nodiscard]] inline auto size() const noexcept { return m_number_of_component; }
        /**
         * @brief Check if morton_index is inside the range of morton indexes of the group_of_particles
         *
         * @param morton_index  morton index
         * @return true if morton_index is inside the group_of_particles
         * @return false otherwise
         */
        [[nodiscard]] inline auto is_inside(std::size_t morton_index) const -> bool
        {
            return ((morton_index >= this->csymbolics().starting_index) &&
                    (morton_index < this->csymbolics().ending_index));
        }
        /**
         * @brief Check if morton_index is below the last morton index of the group_of_particles
         *
         * @param morton_index  morton index
         * @return true if morton_index is below the last morton index of the group_of_particles
         * @return false otherwise
         */
        [[nodiscard]] inline auto is_below(std::size_t morton_index) const -> bool
        {
            return ((morton_index < this->csymbolics().ending_index));
        }
        /**
         * @brief Check if morton_index exists inside the group_of_particles
         *
         * @param morton_index  morton index
         * @return true if morton_index exists inside the group_of_particles
         * @return false otherwise
         */
        [[nodiscard]] inline auto exists(std::size_t morton_index) const -> bool
        {
            return is_inside(morton_index) && (component_index(morton_index) != -1);
        }
        /**
         * @brief return the targeted component
         *
         * @param component_index  the index of the component
         * @return component_type&
         */
        [[nodiscard]] inline auto component(std::size_t component_index) -> component_type&
        {
            assertm(component_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            return m_vector_of_components.at(component_index);
        }

        [[nodiscard]] inline auto component(std::size_t component_index) const -> component_type const&
        {
            assertm(component_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            return m_vector_of_components.at(component_index);
        }

        [[nodiscard]] inline auto ccomponent(std::size_t component_index) const -> component_type const&
        {
            assertm(component_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            return m_vector_of_components.at(component_index);
        }

        [[nodiscard]] inline auto component_iterator(std::size_t index) -> iterator_type
        {
            assertm(index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            auto it = std::begin(m_vector_of_components);
            std::advance(it, index);
            return it;
        }

        [[nodiscard]] inline auto component_iterator(std::size_t index) const -> const_iterator_type
        {
            assertm(index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            auto it = std::cbegin(m_vector_of_components);
            std::advance(it, index);
            return it;
        }

        [[nodiscard]] inline auto ccomponent_iterator(std::size_t cell_index) const -> const_iterator_type
        {
            assertm(cell_index < std::size(m_vector_of_components), "Out of range in group_of_particles.");
            auto it = std::cbegin(m_vector_of_components);
            std::advance(it, cell_index);
            return it;
        }
        /**
         * @brief return the index of the component with the targeted Morton index
         *
         * @tparam MortonIndex
         * @param morton_index  the morton index
         * @return int the index of the component
         */
        template<typename MortonIndex = std::size_t>
        [[nodiscard]] inline auto component_index(MortonIndex morton_index) const -> int
        {
            int idx_left{0};
            int idx_right{static_cast<int>(m_number_of_component) - 1};
            while(idx_left <= idx_right)
            {
                const int idx_middle = (idx_left + idx_right) / 2;
                auto component_index{m_vector_of_components.at(static_cast<std::size_t>(idx_middle)).index()};
                if(component_index == morton_index)
                {
                    return idx_middle;
                }
                if(morton_index < component_index)
                {
                    idx_right = idx_middle - 1;
                }
                else
                {
                    idx_left = idx_middle + 1;
                }
            }
            return -1;
        }

        template<typename MortonIndex = std::size_t>
        [[nodiscard]] inline auto component_index(MortonIndex morton_index, int idx_left, int idx_right) const -> int
        {
            while(idx_left <= idx_right)
            {
                const int idx_middle = (idx_left + idx_right) / 2;
                auto component_index{m_vector_of_components.at(static_cast<std::size_t>(idx_middle)).index()};
                if(component_index == morton_index)
                {
                    return idx_middle;
                }
                if(morton_index < component_index)
                {
                    idx_right = idx_middle - 1;
                }
                else
                {
                    idx_left = idx_middle + 1;
                }
            }
            return -1;
        }
        /**
         * @brief Display the interval of Morton index of the components inside the group
         *
         */
        void print_morton_interval()
        {
            std::cout << "group_of_particles: [" << this->csymbolics().starting_index << ", "
                      << this->csymbolics().ending_index << "]\n";
        }
        /**
         * @brief get the dependencies for the task-based algorithm on the first particle output inside the block
         *
         * @return auto
         */
        auto inline depends_update() { return this->cstorage().cptr_on_output(); }

      private:
        /// Vector of components (cells or leaves) inside the group_of_particles
        block_type m_vector_of_components{};
        /// The storage of the particle or multipole/local values
        storage_type m_components_storage{};
        /// number of components in the group
        const std::size_t m_number_of_component{};
    };

    /// @brief The Symbolics type that stores information about the groupe of leaves
    ///
    /// @tparam P
    template<typename Component, typename Particle>
    struct symbolics_data<group_of_particles<Component, Particle>>
    {
        // using self_type = symbolics_data<group_of_particles<P,D>>;                   
        using self_type = symbolics_data<group_of_particles<Component, Particle>>;
        // the current group type
        using group_type = group_of_particles<Component, Particle>;
        using component_type = Component;
        // the leaf type
        using particle_type = Particle;
        static constexpr std::size_t dimension{particle_type::dimension};
        using iterator_type = typename group_type::iterator_type;
        // the distant group type  // same if source = target or same tree
        using seq_iterator_type =
          std::conditional_t<meta::exist_v<meta::inject<group_type>>, meta::exist_t<meta::inject<group_type>>,
                             std::tuple<iterator_type, group_type>>;
        using group_source_type = typename std::tuple_element_t<1, seq_iterator_type>;
        using iterator_source_type = typename std::tuple_element_t<0, seq_iterator_type>;
        using out_of_block_interaction_type = out_of_block_interaction<iterator_source_type, std::size_t>;
        // storage type to store data in group
        using storage_type = container::particles_block<self_type, particle_type, component_type>;
        // the starting morton index in the group
        std::size_t starting_index{0};
        // the ending morton index in the group
        std::size_t ending_index{0};
        std::size_t number_of_component_in_group{0};
        // number of particles in group
        std::size_t number_of_particles_in_group{0};
        // index of the group
        std::size_t idx_global{0};
        //
        bool is_mine{true};
        // vector storing the out_of_block_interaction structure to handle the outside interactions
        std::vector<out_of_block_interaction_type> outside_interactions{};
        // flagged if the vector is constructed
        bool outside_interactions_exists{false};
        // flagged if the vector is sorted
        bool outside_interactions_sorted{false};
        // #if _OPENMP
        /// the P2P dependencies are set on the pointer on particles container
        // std::array<group_type*, math::pow(dimension, dimension)> group_dependencies{};
        std::vector<group_source_type*> group_dependencies{};
        // #endif

        symbolics_data<group_of_particles<component_type, particle_type>>(std::size_t starting_morton_idx,
                                                            std::size_t ending_morton_idx,
                                                            std::size_t number_of_component,
                                                            std::size_t number_of_particles,
                                                            std::size_t in_index_global, bool in_is_mine)
          : starting_index(starting_morton_idx)
          , ending_index(ending_morton_idx)
          , number_of_component_in_group(number_of_component)
          , number_of_particles_in_group(number_of_particles)
          , idx_global(in_index_global)
          , is_mine(in_is_mine)
        {
        }
    };

}   // namespace scalfmm::component

#endif   // SCALFMM_TREE_GROUP_HPP
