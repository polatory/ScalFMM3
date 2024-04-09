// See LICENCE file at project root
//
#ifndef SCALFMM_TREE_MORTON_CURVE_HPP
#define SCALFMM_TREE_MORTON_CURVE_HPP

#include <scalfmm/container/point.hpp>
#include <scalfmm/utils/math.hpp>
#include <array>
#include <cstddef>

#include "scalfmm/meta/utils.hpp"

/** Provides the corner traversal order of an N dimension hypercube
 *
 * The positions returned are array of booleans. Each boolean tells where
 * to place the element in the binary grid.
 *
 * For instance, in 2D:
 *
 *
 *         __0__ __1__
 *        |     |     |
 *       0|     |  X  |  pos(X) = [true, false]
 *        |_____|_____|
 *        |     |     |
 *       1|     |  Y  |  pos(Y) = [true, true ]
 *        |_____|_____|
 *
 *
 * \tparam dimension The hypercube dimension.
 */
namespace scalfmm::component
{
    template<std::size_t Dim>
    struct z_curve
    {
      public:
        /// Space dimension count
        static constexpr std::size_t dimension = Dim;
        /// Template alias
        template<typename T>
        using position_alias = container::point<T, dimension>;
        /// Position type used
        using position_type = container::point<bool, dimension>;
        /// Position count in the grid
        static constexpr std::size_t pos_count = math::pow(2, dimension);

        constexpr z_curve() { _positions = create_array(); }
        constexpr z_curve(z_curve const&) = default;
        constexpr z_curve(z_curve&&) noexcept = default;
        constexpr inline auto operator=(z_curve const&) -> z_curve& = default;
        constexpr inline auto operator=(z_curve&&) noexcept -> z_curve& = default;
        ~z_curve() = default;

      private:
        /// Array of positions type
        using position_array_t = std::array<position_type, pos_count>;

        /** Creates an array of positions to initialize #_positions */
        constexpr auto create_array() noexcept -> position_array_t
        {
            position_array_t positions;

            for(std::size_t i = 0; i < pos_count; ++i)
            {
                for(std::size_t j = 0, k = i; j < dimension; ++j, k >>= 1)
                {
                    positions[i][j] = k % 2;
                }
            }
            return positions;
        }

        /// Array to cache the positions corresponding to indexes
        position_array_t _positions;

      public:
        /** The position corresponding to an index
         *
         * \param idx The index of the point in the space filling curve
         * \return The position corresponding to the space filling curve index
         */
        [[nodiscard]] constexpr auto position(std::size_t idx) const noexcept -> position_type
        {
            return _positions[idx];
        }

        /** Index in the space filling curve of a boolean position
         *
         * \param p The position
         * \return The space filling curve index corresponding to the position
         */
        [[nodiscard]] constexpr auto index(const position_type& p) const noexcept -> std::size_t
        {
            std::size_t idx = 0;
            for(auto i: p)
            {
                idx <<= 1;
                idx += i;
            }
            return idx;
        }

        /** Index in the space filling curve of a real position relative to the center of the hypercube
         *
         * \param p The position
         * \param center The center of the hypercube
         * \return The space filling curve index corresponding to the position
         */
        template<typename T>
        [[nodiscard]] constexpr auto index(const position_alias<T>& p, const position_alias<T>& center) const noexcept
          -> std::size_t
        {
            std::size_t idx = 0;
            for(std::size_t i = 0; i < dimension; ++i)
            {
                idx <<= 1;
                idx += p[i] > center[i];
            }
            return idx;
        }
    };
}   // namespace scalfmm::component
#endif
