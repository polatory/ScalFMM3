// --------------------------------
// See LICENCE file at project root
// File : core/operators.hpp
// --------------------------------
#ifndef SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP
#define SCALFMM_OPERATORS_COUNT_KERNEL_COUNT_KERNEL_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"

namespace count_kernel
{
    struct execution_infos
    {
        const std::string operator_name{};
        std::size_t particles{0};
        std::size_t number_of_component{0};
        std::size_t call_count{0};
        std::array<std::size_t, 10> multipoles_count{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::array<std::size_t, 10> locals_count{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        friend inline auto operator<<(std::ostream& os, const execution_infos& infos) -> std::ostream&;
    };

    inline auto operator<<(std::ostream& os, const execution_infos& infos) -> std::ostream&
    {
        os << "[\n operator : " << infos.operator_name << "\n  particles : " << infos.particles
           << "\n  call_count : " << infos.call_count << '\n';
        std::size_t level{0};
        for(auto l: infos.multipoles_count)
        {
            std::cout << "  multipoles_count[" << level++ << "] : " << l << '\n';
        }
        level = 0;
        for(auto l: infos.locals_count)
        {
            std::cout << "  locals_count[" << level++ << "] : " << l << '\n';
        }
        std::cout << " ]";
        return os;
    }
    ///
    /// \brief The count_matrix_kernel struct
    ///
    struct count_matrix_kernel
    {
        using map_type = std::unordered_map<std::string, execution_infos>;
        using mapped_type = execution_infos;

        //  static constexpr auto homogeneity_tag{scalfmm::matrix_kernels::homogeneity::homogenous};
        //   static constexpr int km{1};
        //  static constexpr int kn{1};

        //    inline auto separation_criterion() const -> std::size_t { return 1; }
        //     inline auto get(std::string const& key) -> mapped_type& { return m_map.at(key); }

        inline auto print() -> void
        {
            for(auto const& pair: m_map)
            {
                std::cout << pair.second << '\n';
            }
        }

      private:
        map_type m_map{{
          {"p2p_full_mutual", {"p2p_full_mutual"}},
          {"p2p_inner", {"p2p_inner"}},
          {"p2p_remote", {"p2p_remote"}},
        }};
    };
    ///
    /// \brief The count_near_field struct
    ///
    /// An implementation of the near field for the counter kernel
    struct count_near_field
    {
        count_matrix_kernel mat;

        inline auto separation_criterion() const -> std::size_t { return 1; }
        inline count_matrix_kernel matrix_kernel() { return mat; };
        inline auto print() -> void { mat.print(); }
    };
    ///
    /// \brief The count_interpolator struct
    ///
    /// An implementation of the far field for the counter kernel
    // struct count_interpolator
    struct count_far_field
    {
      public:
        using map_type = std::unordered_map<std::string, execution_infos>;
        using map_value_type = typename map_type::value_type;
        using mapped_type = execution_infos;

        //        static constexpr std::size_t dimension{3};
        //        static constexpr std::size_t kn = count_matrix_kernel::kn;
        //        static constexpr std::size_t km = count_matrix_kernel::km;
        inline auto get(std::string const& key) -> mapped_type& { return m_map.at(key); }

        inline auto get(std::string const& key) const -> mapped_type const& { return m_map.at(key); }

        inline auto print() -> void
        {
            for(auto const& pair: m_map)
            {
                std::cout << pair.second << '\n';
            }
        }

        inline auto separation_criterion() const -> std::size_t { return 1; }

        template<typename Cell>
        void apply_multipoles_preprocessing(Cell& current_cell, std::size_t order)
        {
        }
        template<typename Cell>
        void apply_multipoles_postprocessing(Cell& current_cell, std::size_t order)
        {
        }

      private:
        map_type m_map{{
          {"p2m", {"p2m"}},
          {"m2m", {"m2m"}},
          {"m2l", {"m2l"}},
          {"l2l", {"l2l"}},
          {"l2p", {"l2p"}},
          //          {"p2p_full_mutual", {"p2p_full_mutual"}},
          //          {"p2p_inner", {"p2p_inner"}},
          //          {"p2p_remote", {"p2p_remote"}},
        }};
    };

    struct count_fmm_operator
    {
        count_far_field m_far_field;
        count_near_field m_near_field;
        inline count_far_field far_field() { return m_far_field; };
        inline count_near_field near_field() const { return m_near_field; };

        inline auto print() -> void
        {
            //            far_field.print();
            //            near_field.print();
        }
    };

    template<typename Leaf, typename Cell>
    inline void p2m(count_far_field& interp, Leaf const& source_leaf, Cell& target_cell, std::size_t /*order*/)
    {
        auto& infos = interp.get("p2m");
        auto const& target_symb = target_cell.csymbolics();
        auto& multipoles = std::get<0>(target_cell.multipoles());
        multipoles += source_leaf.size();
        infos.call_count++;
        infos.multipoles_count[target_symb.level]++;
        infos.particles += source_leaf.size();
    }

    template<typename Cell>
    inline void l2l(count_far_field& interp, Cell const& parent_cell, std::size_t child_index, Cell& child_cell,
                    std::size_t order, std::size_t tree_level = 2)
    {
        auto& infos = interp.get("l2l");
        auto const& target_symb = child_cell.csymbolics();
        auto& child_locals = std::get<0>(child_cell.locals());
        auto const& current_locals = std::get<0>(parent_cell.clocals());
        child_locals += current_locals;
        infos.call_count++;
        infos.locals_count[target_symb.level]++;
    }

    template<typename Cell>
    inline void m2m(count_far_field& interp, Cell const& child_cell, std::size_t child_index, Cell& parent_cell,
                    std::size_t order, std::size_t tree_level = 2)
    {
        auto& infos = interp.get("m2m");
        auto const& target_symb = parent_cell.csymbolics();
        auto& parent_multipoles = std::get<0>(parent_cell.multipoles());
        auto const& child_multipoles = std::get<0>(child_cell.cmultipoles());
        parent_multipoles += child_multipoles;
        infos.call_count++;
        infos.multipoles_count[target_symb.level]++;
    }

    template<typename Cell>
    inline void m2l(count_far_field& interp, Cell const& source_cell, std::size_t /*neighbor_idx*/, Cell& target_cell,
                    std::size_t /*order*/, std::size_t /*tree_level*/)
    {
        auto& infos = interp.get("m2l");
        auto const& target_symb = target_cell.csymbolics();
        auto const& source_multipoles = std::get<0>(source_cell.cmultipoles());
        auto& target_locals = std::get<0>(target_cell.locals());
        target_locals += source_multipoles;
        infos.locals_count[target_symb.level]++;
    }

    template<typename Cell, typename Leaf, bool ComputeForces = false>
    inline void l2p(count_fmm_operator& fmm_operator, Cell const& source_cell, Leaf& target_leaf, std::size_t /*order*/)
    {
        auto interp = fmm_operator.far_field();
        auto& infos = interp.get("l2p");
        std::get<0>(*scalfmm::container::outputs_begin(target_leaf.particles())) =
          *std::begin(std::get<0>(source_cell.clocals()));
        infos.call_count++;
    }

    template<typename Leaf, typename ContainerOfLeafIterator>
    inline void p2p_full_mutual(count_matrix_kernel const& /*mat*/, Leaf& target_leaf,
                                ContainerOfLeafIterator const& neighbor)
    {
        using value_type = typename Leaf::value_type;
        //      auto& infos = mat.get("p2p_full_mutual");
        value_type nb_val{0};
        std::for_each(std::begin(neighbor), std::end(neighbor),
                      [&target_leaf, &nb_val](auto const& n)
                      {
                          auto neighbor_leaf_output = scalfmm::container::outputs_begin(n->particles());
                          std::get<0>(*neighbor_leaf_output) += static_cast<value_type>(target_leaf.size());
                          nb_val += static_cast<value_type>(n->size());
                      });
        std::get<0>(*scalfmm::container::outputs_begin(target_leaf.particles())) += nb_val;
        //       infos.call_count++;
    }

    template<typename Leaf>
    inline void p2p_inner(count_matrix_kernel const& /*mat*/, Leaf& target_leaf,
                          [[maybe_unused]] const bool mutual = true)
    {
        using value_type = typename Leaf::value_type;
        //        auto& infos = mat.get("p2p_inner");
        std::get<0>(*scalfmm::container::outputs_begin(target_leaf.particles())) +=
          static_cast<value_type>(target_leaf.size());
        //       infos.call_count++;
    }

    template<typename Leaf, typename ContainerOfLeafIterator>
    inline void p2p_remote(count_matrix_kernel const&, Leaf&, ContainerOfLeafIterator const&)
    {
    }
}   // namespace count_kernel

#endif   // SCALFMM_CORE_COUNT_KERNEL_HPP
