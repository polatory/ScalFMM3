// See LICENCE file at project root
// File : box.hpp
// --------------------------------
#ifndef SCALFMM_TREE_BOX_HPP
#define SCALFMM_TREE_BOX_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <scalfmm/tree/morton_curve.hpp>

/* Implements a N dimensions box
 *
 * \author Quentin Khan <quentin.khan@inria.fr>
 *
 * The box is described by two opposite corners : the maximum and
 * the minimum one. All the class transformations maintain this
 * predicate.
 *
 * \tparam value_type Floating number representation.
 * \tparam dimension Space dimension count.
 * \tparam SpaceFillingCurve A templatize implementation of a space filling curve
 *
 */

namespace scalfmm::component
{
    template<class Position, template<std::size_t> class SpaceFillingCurve = z_curve>
    class box
    {
      public:
        /// Position type
        using position_type = Position;
        /// Floating number representation
        using value_type = typename position_type::value_type;
        /// Space dimension
        constexpr static const std::size_t dimension = position_type::dimension;
        /// Space filling curve type
        using space_filling_curve_t = SpaceFillingCurve<dimension>;

      private:
        position_type m_c1;       ///< Minimum corner
        position_type m_c2;       ///< Maximum corner
        position_type m_center;   ///< Center
        space_filling_curve_t m_space_filling_curve;
        std::array<bool, dimension> m_periodicity;   ///< get the periodicity per direction
        bool m_is_periodic;                          ///<  True if a direction is periodic

        /** Rearranges the corners to ensure the maximum-minimum predicate */
        void rearrange_corners()
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                if(m_c1[i] > m_c2[i])
                {
                    std::swap(m_c1[i], m_c2[i]);
                }
            }
            m_center = (m_c1 + m_c2) * 0.5;
        }

      public:
        /// Accessor for the minimum corner
        [[nodiscard]] auto c1() const noexcept -> position_type const& { return m_c1; }
        /// Accessor for the maximum corner
        [[nodiscard]] auto c2() const noexcept -> position_type const& { return m_c2; }

        /** Builds an empty box at the origin */
        box() = default;

        /** Copies an existing box */
        box(const box&) = default;

        /** Copies an other box */
        auto operator=(const box& other) -> box& = default;

        /** Move constructor */
        box(box&&) noexcept = default;

        /** Move assignment */
        auto operator=(box&& other) noexcept -> box& = default;

        /** Destructor */
        ~box() = default;

        /** Builds a cube from the lower corner and its side length
         *
         * \param min_corner The lowest corner
         * \param side_length The cube's side length
         **/
        [[deprecated]] box(const position_type& min_corner, value_type side_length)
          : m_c1(min_corner)
          , m_c2(min_corner)
          , m_space_filling_curve()
          , m_is_periodic(false)
        {
            if(side_length < 0)
            {
                side_length = -side_length;
            }
            for(auto& v: m_periodicity)
            {
                v = false;
            }
            for(auto&& d: m_c2)
            {
                d += side_length;
            }

            m_center = (m_c1 + m_c2) * 0.5;
        }

        /** Builds a cube using its center and width
         *
         * \param width Cube width
         * \param box_center Cube center
         */
        box(value_type width, position_type const& box_center)
          : m_c1(box_center)
          , m_c2(box_center)
          , m_center(box_center)
          , m_space_filling_curve()
          , m_is_periodic(false)
        {
            if(width < 0)
            {
                width = -width;
            }
            for(auto& v: m_periodicity)
            {
                v = false;
            }
            value_type radius = width / 2;

            for(auto&& d: m_c1)
            {
                d -= radius;
            }

            for(auto&& d: m_c2)
            {
                d += radius;
            }
        }

        /** Builds a box from two corners
         *
         * The maximum and minimum corners are deduced from the given corners.
         *
         * \param corner_1 The first corner.
         * \param corner_2 The second corner.
         */
        box(const position_type& corner_1, const position_type& corner_2)
          : m_c1(corner_1)
          , m_c2(corner_2)
          , m_space_filling_curve()
          , m_is_periodic(false)
        {
            for(auto& v: m_periodicity)
            {
                v = false;
            }

            rearrange_corners();
        }

        /** Changes the box corners
         *
         * The maximum and minimum corners are deduced from the given corners.
         *
         * \param corner_1 The first corner.
         * \param corner_2 The second corner.
         */
        void set(const position_type& corner_1, const position_type& corner_2)
        {
            m_c1 = corner_1;
            m_c2 = corner_2;

            rearrange_corners();
        }

        /** Checks whether a position is within the box bounds
         *
         * \param p The position to check.
         */
        [[nodiscard]] auto contains(const position_type& p) const -> bool
        {
            for(std::size_t i = 0; i < dimension; ++i)
            {
                if(p[i] < m_c1[i] || p[i] > m_c2[i])
                {
                    return false;
                }
            }
            return true;
        }

        /** Checks whether an object's position is within the box bounds
         *
         * The object must implement a `position_type position() const;` method.
         *
         * \tparam T The object type.
         * \param obj The object which position to check.
         */
        template<class T>
        [[nodiscard]] auto contains(const T& obj) const -> bool
        {
            return contains(obj.position());
        }

        /** Accessor for the box center */
        [[nodiscard]] auto center() const -> position_type const& { return m_center; }

        /** Accessor for the box center */
        template<typename Int>
        [[nodiscard]] auto extended_center(Int added_levels) const -> position_type const
        {
            if(m_is_periodic)
            {
                const auto original_width = this->width(0);
                const auto offset = 0.5 * (this->extended_width(added_levels) - original_width);
                return m_center + position_type(offset);
            }
            else
            {
                std::clog << "Warning extended_center called but not periodic simulation.\n";
                return m_center;
            }
        }

        /** Accessor for the box corners
         *
         * The corners are numbered using a space filling curve.
         *
         * \param idx The corner index.
         * \return The idx'th corner.
         */
        [[nodiscard]] auto corner(std::size_t idx) const -> position_type
        {
            position_type c;
            std::size_t i = 0;
            for(bool choice: m_space_filling_curve.position(idx))
            {
                c[i] = choice ? m_c2[i] : m_c1[i];
                ++i;
            }
            return c;
        }

        /** Setter for the corners
         *
         * Moves a corner to a new position and modifies the relevant other
         * ones. The corners are numbered using a space filling curve.
         *
         * \param idx The moved corner index.
         * \param pos The new corner position.
         */
        void corner(std::size_t idx, const position_type& pos)
        {
            std::size_t i = 0;
            for(bool choice: m_space_filling_curve.position(idx))
            {
                if(choice)
                {
                    m_c2[i] = pos[i];
                }
                else
                {
                    m_c1[i] = pos[i];
                }
                ++i;
            }
            rearrange_corners();
        }
#ifdef SCALFMM_BUILD_PBC
  
        ///
        /// \brief set the periodicity in the direction dir
        /// \param dir  direction
        /// \param per  true if the direction dir is periodic otherwise false
        ///
        void set_periodicity(int dir, bool per)
        {
            m_periodicity[dir] = per;
            m_is_periodic = m_is_periodic || per;
        }

        ///
        /// \brief set the periodicity in all directions
        /// \param pbl  a vector of boolean that specifies if the direction is periodic or not
        ///
        template<typename Vector>
        void set_periodicity(const Vector& pbl)
        {
            for(std::size_t d = 0; d < pbl.size(); ++d)
                set_periodicity(d, pbl[d]);
        }
#endif
        ///
        /// \brief get_periodicity return the array of periodic direction
        ///
        ///
        auto get_periodicity() const -> std::array<bool, dimension> { return m_periodicity; }
        ///
        /// \brief is_periodic tell if there is a periodic direction
        ///
        ///
        auto is_periodic() const noexcept -> bool { return m_is_periodic; }

        /** Returns the width for given dimension */
        [[nodiscard]] auto width(std::size_t dim) const noexcept -> decltype(std::abs(m_c2[dim] - m_c1[dim]))
        {
            return std::abs(m_c2[dim] - m_c1[dim]);
        }
        /** Returns the extended width for periodic simulation for  added_levels n*/
        template<typename Int>
        [[nodiscard]] auto extended_width(Int added_levels) const noexcept -> value_type
        {
            auto width = this->width(0);

            if(m_is_periodic || added_levels > 0)
            {
                width *= ((4) << added_levels);
            }
            else
            {
                std::clog << "Warning extended_width called but not periodic simulation.\n";
            }
            return width;
        }
        /** Sums the corners of two boxes */
        auto operator+(const box& other) const -> box { return box(m_c1 + other.m_c1, m_c2 + other.m_c2); }

        /** Tests two boxes equality */
        auto operator==(const box& other) const -> bool { return c1() == other.c1() && c2() == other.c2(); }
        /** Tests two boxes inequality */
        auto operator!=(const box& other) const -> bool { return !this->operator==(other); }

        friend auto operator<<(std::ostream& os, const box& box) -> std::ostream&
        {
            os << " [" << box.c1() << "," << box.c2() << "] ; periodicity: ";
            if(box.is_periodic())
            {
                os << std::boolalpha << " ( ";
                const auto& per = box.get_periodicity();
                for(std::size_t i = 0; i < dimension - 1; ++i)
                {
                    os << per[i] << ", ";
                }
                os << per[dimension - 1] << ")";
            }
            else
            {
                os << std::boolalpha << " none ";
            }
            return os;
        }
    };
}   // namespace scalfmm::component
#endif
