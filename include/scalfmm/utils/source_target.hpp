#ifndef SCALFMM_UTILS_SOURCE_TARGET_HPP
#define SCALFMM_UTILS_SOURCE_TARGET_HPP

#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/for_each.hpp>

namespace scalfmm::utils
{

    /**
     * @brief Construct the geometric bounding box of box1 and box2
     *
     * @param[in] box1 first box containing particles (sources)
     * @param[in] box2 second box containing particles (targets)
     * @return the box containing all particles (sources and targets)
     */
    template<typename box_type>
    inline auto bounding_box(box_type const& box1, box_type const& box2) -> box_type
    {
        using position_type = typename box_type::position_type;
        using value_type = typename position_type::value_type;
        constexpr static const std::size_t dimension = position_type::dimension;

        const auto& box1_c1 = box1.c1();
        const auto& box1_c2 = box1.c2();
        const auto& box2_c1 = box2.c1();
        const auto& box2_c2 = box2.c2();
        position_type c1{}, c2{};
        for(std::size_t i = 0; i < dimension; ++i)
        {
            c1.at(i) = std::min(box1_c1.at(i), box2_c1.at(i));
            c2.at(i) = std::max(box1_c2.at(i), box2_c2.at(i));
        }

        auto diff = c2 - c1;
        value_type length{diff.max()};
        c2 = c1 + length;

        return box_type(c1, c2);
    }

    /**
     * @brief Merge two containers of particles
     *
     * @param container1 first container
     * @param container2 second container
     * @return the merge of the two containers
     */
    template<typename Container_type>
    auto merge(Container_type const& container1, Container_type const& container2) -> Container_type
    {
        auto size = container1.size() + container2.size();
        Container_type output(size);
        std::size_t idx = 0;
        for(std::size_t i = 0; i < container1.size(); ++i, ++idx)
        {
            output.insert_particle(idx, container1.particle(i));
        }
        for(std::size_t i{0}; i < container2.size(); ++i, ++idx)
        {
            output.insert_particle(idx, container2.particle(i));
        }
        return output;
    }

    template<typename Treetype, typename Index_type>
    inline auto find_cell_group(Treetype const& tree, Index_type index, int& level)
    {
        std::size_t id_group{0};
    }

}   // namespace scalfmm::utils
#endif
