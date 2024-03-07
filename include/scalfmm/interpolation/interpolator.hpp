// --------------------------------
// See LICENCE file at project root
// File : interpolation/interpolator.hpp
// --------------------------------
#ifndef SCALFMM_INTERPOLATION_INTERPOLATOR_HPP
#define SCALFMM_INTERPOLATION_INTERPOLATOR_HPP

#include <algorithm>
#include <any>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor/xvectorize.hpp>

#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/builders.hpp"
#include "scalfmm/interpolation/mapping.hpp"
#include "scalfmm/interpolation/permutations.hpp"
#include "scalfmm/matrix_kernels/mk_common.hpp"
#include "scalfmm/memory/storage.hpp"
#include "scalfmm/meta/const_functions.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/options/options.hpp"
#include "scalfmm/utils/low_rank.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"
#include "xsimd/config/xsimd_config.hpp"
#include "xsimd/memory/xsimd_aligned_allocator.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xtensor_config.hpp"
#include "xtensor/xview.hpp"

#include <xtensor/xnpy.hpp>

#include "scalfmm/utils/io_helpers.hpp"
using namespace scalfmm::io;

namespace scalfmm::interpolation
{
    template<typename Derived>
    struct interpolator_traits;

    namespace impl
    {

        template<typename Derived, typename Enable = void>
        class interpolator
        {
            static_assert(std::is_floating_point_v<typename interpolator_traits<Derived>::value_type>,
                          "interpolator ValueType needs to be floating point.");
        };

        template<typename Derived>
        struct alignas(XTENSOR_FIXED_ALIGN) interpolator<
          Derived,
          typename std::enable_if_t<std::is_floating_point_v<typename interpolator_traits<Derived>::value_type>>>
        {
          private:
            struct empty
            {
            };

          public:
            using derived_type = Derived;
            using value_type = typename interpolator_traits<Derived>::value_type;
            using size_type = std::size_t;

            static constexpr std::size_t dimension = interpolator_traits<Derived>::dimension;

            using array_type = xt::xarray<value_type>;
            using array_shape_type = typename array_type::shape_type;

            template<std::size_t d>
            using tensor_type = xt::xtensor<value_type, d>;
            template<std::size_t d>
            using tensor_shape_type = typename tensor_type<d>::shape_type;
            using grid_permutations_type = std::conditional_t<(dimension > 3), empty, xt::xtensor<std::size_t, 2>>;
            using settings = typename interpolator_traits<derived_type>::settings;

            interpolator() = delete;
            interpolator(interpolator const& other) = delete;
            interpolator(interpolator&&) noexcept = delete;
            auto operator=(interpolator const&) -> interpolator& = delete;
            auto operator=(interpolator&&) noexcept -> interpolator& = delete;
            ~interpolator() = default;

            /// @brief constructor of the interpolator
            ///
            /// @param order number of terms of the 1d expansion ()
            /// @param tree_height height of the tree
            /// @param root_cell_width width of the simulation box
            /// @param cell_width_extension  width of the extension of the cell
            /// @param late_init if true the initialization is done later
            interpolator(size_type order, size_type tree_height, value_type root_cell_width,
                         value_type cell_width_extension, bool late_init = false)
              : m_order(order)
              , m_nnodes(meta::pow(order, dimension))
              , m_tree_height(tree_height)
              , m_root_cell_width(root_cell_width)
              , m_cell_width_extension(cell_width_extension)
              , m_grid_permutations(this->generate_grid_permutations(order))
            {
                if(late_init == false)
                {
                    this->initialize();
                }
            }

            /// @brief constructor of the interpolator
            ///
            /// @param late_init
            /// @param order number of terms of the 1d expansion ()
            /// @param tree_height height of the tree
            /// @param root_cell_width width of the simulation box
            /// @param late_init if true the initialization is done later
            interpolator(size_type order, size_type tree_height, value_type root_cell_width, bool late_init = false)
              : interpolator(order, tree_height, root_cell_width, value_type(0.), late_init)
            {
            }
            /// @brief return the number of 1d points (number of terms in the 1d-expansion)
            [[nodiscard]] inline auto order() const noexcept -> std::size_t { return m_order; }
            /// @brief eturn the number of pointsin de dimension grid
            [[nodiscard]] inline auto nnodes() const noexcept { return m_nnodes; }
            /// @brief
            [[nodiscard]] inline auto cell_width_extension() const noexcept -> value_type
            {
                return m_cell_width_extension;
            }
            /// @brief
            [[nodiscard]] inline auto interpolator_tensor() noexcept -> array_type&
            {
                return m_child_parent_interpolators;
            }

            [[nodiscard]] inline auto interpolator_tensor() const noexcept -> array_type const&
            {
                return m_child_parent_interpolators;
            }

            [[nodiscard]] inline auto cinterpolator_tensor() const noexcept -> array_type const&
            {
                return m_child_parent_interpolators;
            }
            /// @brief return the roots of the polynomial
            [[nodiscard]] inline auto roots() const -> array_type { return this->derived_cast().roots_impl(); }

            template<typename ComputationType>
            [[nodiscard]] inline auto polynomials(ComputationType x, std::size_t n) const -> ComputationType
            {
                return this->derived_cast().polynomials_impl(x, n);
            }
            template<typename VectorType, typename ComputationType, std::size_t Dim>
            inline auto fill_all_polynomials(VectorType& all_poly, container::point<ComputationType, Dim>& x,
                                             std::size_t order) const -> void
            {
                this->derived_cast().fill_all_polynomials_impl(all_poly, x, order);
            }
            template<typename ComputationType>
            [[nodiscard]] inline auto derivative(ComputationType x, std::size_t n) const
            {
                return this->derived_cast().derivative_impl(x, n);
            }

            [[nodiscard]] inline auto grid_permutations() const -> xt::xtensor<std::size_t, 2> const&
            {
                return m_grid_permutations;
            }

          private:
            /// @brief This function pre-calculates all the child-parent interpolators required for M2M and L2L passes.
            ///
            /// This function pre-calculates all the child-parent interpolators required for M2M and L2L passes
            /// for all levels that are not leaf levels. If there is no cell width extension, the same set of
            /// child-parent interpolators is used for all levels. However, if there is a cell width extension, the
            /// child-parent interpolators must be recalculated for each level because the ratio of "child cell width
            /// over parent cell width" is not the same.
            ///
            /// @warning(homogeneity not yet supported when cell_width_extension > 0)
            ///
            void set_child_parent_interpolator()
            {
                using point_type = scalfmm::container::point<value_type, dimension>;

                // Get the number of child cells (depending on to the dimension)
                constexpr auto childs_number = meta::pow(2, dimension);

                // Set number of non-leaf ios that actually need to be computed
                // Maybe always 3 ?
                size_type reduced_tree_height{std::min(size_type{3}, m_tree_height)};   // = 2 + nb of computed nl ios

                // If there is no cell extension, the same set of child-parent interpolator is used for each level
                if(m_cell_width_extension > value_type(0.))
                {
                    reduced_tree_height = m_tree_height;   // compute 1 non-leaf io per level
                }

                // Init non-leaf interpolators
                constexpr value_type half{0.5};
                constexpr value_type two{2.};
                value_type cell_width{m_root_cell_width * half};   // at level 1
                cell_width *= half;                                // at level 2

                // Allocate memory for child-parent interpolator
                // expliquer les differents indices du tenseurs
                m_child_parent_interpolators.resize({reduced_tree_height, childs_number, dimension, m_order * m_order});

                // Generate 1-D ones combinations for relative childs positions
                std::vector<value_type> to_adapt(childs_number * dimension);
                using to_adapt_difference_type =
                  typename std::iterator_traits<typename std::vector<value_type>::iterator>::difference_type;

                for(std::size_t i = 0; i < childs_number; ++i)
                {
                    std::array<value_type, dimension> binary{};
                    auto dec{i};
                    auto count{dimension - 1};

                    while(dec > 0)
                    {
                        binary.at(count) = static_cast<value_type>(dec % 2);
                        dec /= 2;
                        count--;
                    }
                    std::replace_if(binary.begin(), binary.end(),
                                    std::bind(std::equal_to<value_type>(), std::placeholders::_1, value_type(0.)),
                                    value_type(-1.));
                    std::copy(binary.begin(), binary.end(),
                              to_adapt.begin() + static_cast<to_adapt_difference_type>(i * dimension));
                }

                const array_shape_type relative_childs_centers_shape({childs_number, dimension});
                array_type relative_childs_centers(relative_childs_centers_shape);

                // Adapting vector of relative positions
                relative_childs_centers = xt::adapt(to_adapt, relative_childs_centers_shape);

                // Get the 1D interpolation roots (on the reference interval [-1, 1])
                array_type inter_roots(roots());

                // Temporary array to store the interpolation roots in the child cells
                const array_shape_type child_coords_shape({dimension, m_order});
                array_type child_coords(child_coords_shape);

                // Loop on all levels that are not at leaf level
                for(std::size_t level = 2; level < reduced_tree_height; ++level)
                {
                    // Ratio of extended cell widths at the current level (definition: child ext / parent ext)
                    const value_type extended_cell_ratio =
                      (cell_width * half + m_cell_width_extension) / (cell_width + m_cell_width_extension);

                    // Child cell width
                    const value_type childs_width{two * extended_cell_ratio};

                    // Scale factor for the relative child centers (taking into account the extension)
                    const value_type scale_factor = value_type(1.) - extended_cell_ratio;

                    // Loop on the child cells
                    for(std::size_t child_index = 0; child_index < childs_number; ++child_index)
                    {
                        // Set relative child center (taking into account the extension)
                        point_type child_center{}, l_pos{}, g_pos{};
                        for(std::size_t d = 0; d < dimension; ++d)
                        {
                            child_center[d] = relative_childs_centers(child_index, d) * scale_factor;
                        }

                        // Set polynomial roots in the child cell (using a "local to global" mapping)
                        const map_loc_glob<point_type> mapper_lg(child_center, point_type(childs_width));

                        for(std::size_t o = 0; o < m_order; ++o)
                        {
                            l_pos = point_type(inter_roots[o]);
                            mapper_lg(l_pos, g_pos);
                            for(std::size_t d = 0; d < dimension; ++d)
                            {
                                child_coords(d, o) = g_pos[d];
                            }
                        }

                        // Assemble the corresponding interpolator
                        for(std::size_t o1 = 0; o1 < m_order; ++o1)
                        {
                            for(std::size_t o2 = 0; o2 < m_order; ++o2)
                            {
                                for(std::size_t d = 0; d < dimension; ++d)
                                {
                                    // ajouter une variable pour l'acces
                                    m_child_parent_interpolators(level, child_index, d, o1 * m_order + o2) =
                                      polynomials(child_coords(d, o1), o2);
                                }
                            }
                        }
                    }

                    // Update cell width for the next level
                    cell_width *= half;
                }
            }

            /// @brief The former version of "set_child_parent_interpolator"
            ///
            /// @warning May produce unexpected results when tree_height > 3 and cell_width_extension > 0
            ///
            void set_child_parent_interpolator_bis()
            {
                using namespace xt::placeholders;

                constexpr auto childs_number = meta::pow(2, dimension);

                // Set number of non-leaf ios that actually need to be computed
                size_type reduced_tree_height{0};   // = 2 + nb of computed nl ios

                if(m_cell_width_extension == 0.)
                {                                                                  // if no cell extension, then ...
                    reduced_tree_height = std::min(size_type{3}, m_tree_height);   // cmp only 1 non-leaf io
                }
                else
                {
                    reduced_tree_height = m_tree_height;   // cmp 1 non-leaf io per level
                }

                // Init non-leaf interpolators
                constexpr value_type half{0.5};
                constexpr value_type two{2.};
                value_type cell_width = m_root_cell_width * half;   // at level 1
                cell_width *= half;                                 // at level 2

                // Allocate child-parent interpolator
                m_child_parent_interpolators.resize({reduced_tree_height, childs_number, dimension, m_order * m_order});

                // Allocate childs coords
                const tensor_shape_type<3> child_coords_shape{childs_number, dimension, m_order};
                tensor_type<3> child_coords(child_coords_shape);

                array_type childs_centers;
                array_type extended_cell_ratios;

                const array_shape_type child_sized_shape({childs_number, dimension});

                // Resizing centers and cell ratios to n_childs*dimension
                childs_centers.resize(child_sized_shape);
                extended_cell_ratios.resize(child_sized_shape);

                // Compute ratio of extended cell widths (definition: child ext / parent ext)
                extended_cell_ratios =
                  xt::full_like(extended_cell_ratios,
                                (cell_width * half + m_cell_width_extension) / (cell_width + m_cell_width_extension));
                // Child width cell
                array_type childs_width(extended_cell_ratios * two);

                // Generate 1-D ones combinations for relative childs positions
                std::vector<value_type> to_adapt(childs_number * dimension);
                using to_adapt_difference_type =
                  typename std::iterator_traits<typename std::vector<value_type>::iterator>::difference_type;

                for(std::size_t i = 0; i < childs_number; ++i)
                {
                    std::array<value_type, dimension> binary{};
                    auto dec{i};
                    auto count{dimension - 1};

                    while(dec > 0)
                    {
                        binary.at(count) = static_cast<value_type>(dec % 2);
                        dec /= 2;
                        count--;
                    }
                    std::replace_if(binary.begin(), binary.end(),
                                    std::bind(std::equal_to<value_type>(), std::placeholders::_1, value_type(0.)),
                                    value_type(-1.));
                    std::copy(binary.begin(), binary.end(),
                              to_adapt.begin() + static_cast<to_adapt_difference_type>(i * dimension));
                }

                const array_shape_type relative_childs_centers_shape({childs_number, dimension});
                array_type relative_childs_centers(relative_childs_centers_shape);

                // Adapting vector of positions
                relative_childs_centers = xt::adapt(to_adapt, relative_childs_centers_shape);

                // Translate center if cell widths are extended
                const auto scale_factor = xt::ones_like(extended_cell_ratios) - extended_cell_ratios;

                // Update child centers
                childs_centers = relative_childs_centers * scale_factor;

                // Setting polynomial roots
                const map_loc_glob<array_type> mapper_lg(childs_centers, childs_width);

                // Get roots
                array_type inter_roots(roots());

                // global positions
                array_type g_pos(child_sized_shape);
                array_shape_type to_reshape = {childs_number, dimension, m_order};

                for(std::size_t level = 2; level < reduced_tree_height; ++level)
                {
                    for(std::size_t o = 0; o < m_order; ++o)
                    {
                        // Broadcasting roots
                        auto l_pos = xt::ones<value_type>(relative_childs_centers_shape) * inter_roots[o];

                        // Mapping to global pos
                        mapper_lg(l_pos, g_pos);
                        // Update child_coords
                        auto v = xt::view(child_coords, xt::all(), xt::all(), o);
                        v = g_pos;
                    }

                    for(std::size_t o1 = 0; o1 < m_order; ++o1)
                    {
                        for(std::size_t o2 = 0; o2 < m_order; ++o2)
                        {
                            auto v_interpolators =
                              xt::view(m_child_parent_interpolators, level, xt::all(), xt::all(), o1 * m_order + o2);
                            auto v_child_coords = xt::view(child_coords, xt::all(), xt::all(), o1);

                            auto call_polynomials = [this](value_type x, std::size_t n) -> value_type
                            { return polynomials(x, n); };

                            auto v_get_polynomials = xt::vectorize(call_polynomials);
                            v_interpolators = v_get_polynomials(v_child_coords, o2);
                        }
                    }
                }
            }

            inline auto generate_grid_permutations(std::size_t order)
            {
                grid_permutations_type perms{};

                if constexpr(dimension == 2)
                {
                    perms.resize({dimension, math::pow(order, dimension)});
                    std::size_t flat{0};
                    for(std::size_t i = 0; i < order; ++i)
                    {
                        for(std::size_t j = 0; j < order; ++j)
                        {
                            perms.at(0, flat) = i * order + j;
                            perms.at(1, flat) = j * order + i;
                            flat++;
                        }
                    }
                }
                else if constexpr(dimension == 3)
                {
                    perms.resize({dimension, math::pow(order, dimension)});
                    std::size_t flat{0};
                    for(std::size_t i = 0; i < order; ++i)
                    {
                        for(std::size_t j = 0; j < order; ++j)
                        {
                            for(std::size_t k = 0; k < order; ++k)
                            {
                                perms.at(0, flat) = i * order * order + j * order + k;
                                perms.at(1, flat) = k * order * order + i * order + j;
                                perms.at(2, flat) = j * order * order + k * order + i;
                                flat++;
                            }
                        }
                    }
                }
                return perms;
            }

          protected:
            inline auto initialize() -> void { set_child_parent_interpolator(); }

            [[nodiscard]] inline auto derived_cast() & noexcept -> derived_type&
            {
                return *static_cast<derived_type*>(this);
            }

            [[nodiscard]] inline auto derived_cast() const& noexcept -> derived_type const&
            {
                return *static_cast<const derived_type*>(this);
            }

            [[nodiscard]] inline auto derived_cast() && noexcept -> derived_type
            {
                return *static_cast<derived_type*>(this);
            }

          private:
            const size_type m_order{};                      ///< number of terms of the expansion (1d)
            const size_type m_nnodes{};                     ///< number of modes m_order^dimension
            const size_type m_tree_height{};                ///< height of the tree
            const value_type m_root_cell_width{};           ///< width of the simulation box
            const value_type m_cell_width_extension{};      ///< width of the extension of the cell
            array_type m_child_parent_interpolators{};      ///<
            grid_permutations_type m_grid_permutations{};   ///<
        };

        /**
         * @warning Cell width extension is not yet supported for homogeneous kernels in the latest version of ScalFMM!
         */
        template<typename Derived>
        struct m2l_handler
        {
          private:
            struct empty
            {
            };

          public:
            using derived_type = Derived;
            using value_type = typename interpolator_traits<derived_type>::value_type;
            using size_type = std::size_t;

            static constexpr std::size_t dimension = interpolator_traits<derived_type>::dimension;

            using matrix_kernel_type = typename interpolator_traits<derived_type>::matrix_kernel_type;

            using settings = typename interpolator_traits<derived_type>::settings;

            static constexpr std::size_t kn = matrix_kernel_type::kn;
            static constexpr std::size_t km = matrix_kernel_type::km;
            static constexpr auto separation_criterion = matrix_kernel_type::separation_criterion;
            static constexpr auto homogeneity_tag = matrix_kernel_type::homogeneity_tag;
            static constexpr auto symmetry_tag = matrix_kernel_type::symmetry_tag;
            static constexpr auto enable_symmetries = interpolator_traits<derived_type>::enable_symmetries;
            static constexpr std::size_t max_number_of_cell{7};
            static constexpr bool symmetry_support{
              (symmetry_tag == matrix_kernels::symmetry::symmetric && (enable_symmetries == true) && (dimension < 4))};

            using scale_factor_type = typename matrix_kernel_type::template vector_type<value_type>;
            using sym_permutations_type = std::conditional_t<symmetry_support, xt::xarray<int>, empty>;
            using k_indices_type = std::conditional_t<symmetry_support, std::vector<std::size_t>, empty>;

            using array_type = xt::xarray<value_type>;
            using array_shape_type = typename array_type::shape_type;

            template<std::size_t d>
            using tensor_type = xt::xtensor<value_type, d>;
            template<std::size_t d>
            using tensor_shape_type = typename tensor_type<d>::shape_type;

            using storage_type = typename interpolator_traits<derived_type>::storage_type;
            using buffer_value_type = typename interpolator_traits<derived_type>::buffer_value_type;
            using buffer_inner_type = typename interpolator_traits<derived_type>::buffer_inner_type;
            using k_tensor_type = std::conditional_t<std::is_same_v<settings, options::low_rank_>,
                                                     std::tuple<xt::xtensor<value_type, 2>, xt::xtensor<value_type, 2>>,
                                                     typename interpolator_traits<derived_type>::k_tensor_type>;
            using interaction_matrix_type = xt::xtensor_fixed<k_tensor_type, xt::xshape<kn, km>>;
            using buffer_shape_type = typename interpolator_traits<derived_type>::buffer_shape_type;
            using buffer_type = typename interpolator_traits<derived_type>::buffer_type;
            using multipoles_inner_type =
              typename memory::storage_traits<typename storage_type::multipoles_storage_type>::inner_type;
            using locals_inner_type =
              typename memory::storage_traits<typename storage_type::locals_storage_type>::inner_type;

            m2l_handler() = delete;
            m2l_handler(m2l_handler const& other) = delete;
            m2l_handler(m2l_handler&&) noexcept = delete;
            auto operator=(m2l_handler const&) -> m2l_handler& = delete;
            auto operator=(m2l_handler&&) noexcept -> m2l_handler& = delete;
            ~m2l_handler() = default;

            m2l_handler(matrix_kernel_type const& far_field, array_type roots, size_type tree_height = 3,
                        value_type root_cell_width = value_type(1.), value_type cell_width_extension = value_type(0.),
                        bool late_init = false)
              : m_far_field(far_field)
              , m_m2l_interactions(math::pow(max_number_of_cell, dimension))
              , m_nnodes(meta::pow(roots.size(), dimension))
              , m_order(roots.size())
              , m_roots(roots)
              , m_epsilon(std::pow(value_type(10.), -value_type(roots.size())))
              , m_cell_width_extension(cell_width_extension)
            {
                if((cell_width_extension > 0) && (homogeneity_tag == matrix_kernels::homogeneity::homogenous))
                {
                    throw std::runtime_error("Cell width extension is not yet supported for homogeneous kernels in the "
                                             "latest version of ScalFMM!");
                }

                if(late_init == false)
                {
                    this->initialize(roots.size(), root_cell_width, tree_height);
                }

                if constexpr(symmetry_support)
                {
                    std::tie(m_sym_permutations, m_k_indices) = get_permutations_and_indices<dimension>(
                      roots.size(), meta::pow(roots.size(), dimension), this->m2l_interactions());
                }
            }

            [[nodiscard]] inline auto weights() const noexcept -> array_type const& { return m_weights_of_roots; }

            [[nodiscard]] inline auto epsilon() noexcept -> value_type& { return m_epsilon; }
            [[nodiscard]] inline auto epsilon() const noexcept -> value_type { return m_epsilon; }
            // member function to get the index of K corresponding to the interaction index in the interaction matrice
            // vector.
            [[nodiscard]] inline auto symmetry_k_index(std::size_t neighbor_idx) const -> std::size_t
            {
                return m_k_indices.at(neighbor_idx);
            }

            template<typename TensorOrViewX, typename TensorOrViewY>
            [[nodiscard]] inline auto generate_matrix_k(TensorOrViewX&& X, TensorOrViewY&& Y, std::size_t n,
                                                        std::size_t m, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                !decltype(meta::sig_gen_k_f(std::declval<derived_type>(), X, Y, n, m, thread_id))::value, k_tensor_type>

            {
                if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    auto const& matrix_kernel{this->matrix_kernel()};

                    auto n_d = math::pow(m_order, dimension);

                    k_tensor_type K(std::vector(2, n_d));
                    auto flat_x = xt::flatten(X);
                    auto flat_y = xt::flatten(Y);
                    // TODO : SIMD!

                    for(std::size_t i{0}; i < n_d; ++i)
                    {
                        for(std::size_t j{0}; j < n_d; ++j)
                        {
                            K.at(i, j) = matrix_kernel.evaluate(flat_x.at(i), flat_y.at(j)).at(n * km + m);
                        }
                    }
                    // std::cout << cpp_tools::colors::cyan;
                    // std::cout << K << std::endl;
                    // std::cout << cpp_tools::colors::reset;
                    // xt::dump_npy("interaction_matrix_non_homogenous_with_ext.npy",K);
                    return K;
                }
                else if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    auto const& matrix_kernel{this->matrix_kernel()};
                    return low_rank::generate(matrix_kernel, std::forward<TensorOrViewX>(X),
                                              std::forward<TensorOrViewY>(Y), this->weights(), this->epsilon(), n, m);
                }
                else
                {
                    throw std::runtime_error("Missing generate_matrix_k function!");
                }
            }
            template<typename TensorOrViewX, typename TensorOrViewY>
            [[nodiscard]] inline auto generate_matrix_k(TensorOrViewX&& X, TensorOrViewY&& Y, std::size_t n,
                                                        std::size_t m, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                decltype(meta::sig_gen_k_f(std::declval<derived_type>(), X, Y, n, m, thread_id))::value, k_tensor_type>
            {
                return this->derived_cast().generate_matrix_k_impl(std::forward<TensorOrViewX>(X),
                                                                   std::forward<TensorOrViewX>(Y), n, m, thread_id);
            }

            [[nodiscard]] inline auto interactions_matrices() noexcept
              -> std::vector<interaction_matrix_type, XTENSOR_DEFAULT_ALLOCATOR(interaction_matrix_type)>&
            {
                return m_interactions_matrices;
            }

            [[nodiscard]] inline auto interactions_matrices() const noexcept
              -> std::vector<interaction_matrix_type> const&
            {
                return m_interactions_matrices;
            }

            [[nodiscard]] inline auto cinteractions_matrices() const noexcept
              -> std::vector<interaction_matrix_type> const&
            {
                return m_interactions_matrices;
            }

            [[nodiscard]] inline auto m2l_interactions() const noexcept { return m_m2l_interactions; }

            [[nodiscard]] inline auto sym_permutations() const -> sym_permutations_type const&
            {
                return m_sym_permutations;
            }

            [[nodiscard]] inline auto matrix_kernel() const noexcept -> matrix_kernel_type const&
            {
                return m_far_field;
            }

            template<typename D = derived_type>
            [[nodiscard]] auto buffer_initialization() const
              -> std::enable_if_t<decltype(meta::sig_buffer_init_f(std::declval<D>()))::value, buffer_type>
            {
                return this->derived_cast().buffer_initialization_impl();
            }
            /**
             * @brief Initialize the buffer to aggregate the multipoles and teh locals when the kernel is symmetric
             *
             *  if the kernel is non symmetric we have an empty buffers otherwise the element of the buffer
             *   is a tensor of size (number of multipole associated to the current symmetry, the number of nodes)
             *
             * @tparam D
             * @return std::enable_if_t<!decltype(meta::sig_buffer_init_f(std::declval<D>()))::value, buffer_type>
             */
            template<typename D = derived_type>
            [[nodiscard]] auto buffer_initialization() const
              -> std::enable_if_t<!decltype(meta::sig_buffer_init_f(std::declval<D>()))::value, buffer_type>
            {
                std::vector<std::size_t> shape(2, m_nnodes);
                if constexpr(enable_symmetries)
                {
                    shape[1] = interpolation::largest_number_permutation<dimension>();
                }
                return buffer_type(buffer_shape_type{}, buffer_inner_type(shape, 0.));
            }

            /**
             * @brief Reset the buffer by calling buffer_reset_impl (specialization)
             *
             * @tparam D
             * @param buffers
             */
            template<typename D = derived_type>
            inline auto buffer_reset(buffer_type& buffers) const
              -> std::enable_if_t<decltype(meta::sig_buffer_reset_f(std::declval<D>(), buffers))::value, void>
            {
                this->derived_cast().buffer_reset_impl(buffers);
            }
            /**
             * @brief Reset the buffer, generic function
             *
             * @tparam D
             * @param buffers
             */
            template<typename D = derived_type>
            inline auto buffer_reset(buffer_type& buffers) const
              -> std::enable_if_t<!decltype(meta::sig_buffer_reset_f(std::declval<D>(), buffers))::value, void>
            {
                if constexpr(symmetry_support)
                {
                    for(std::size_t n{0}; n < 2; ++n)
                    {
                        buffers.at(n).fill(buffer_value_type(0.));
                    }
                }
            }

            template<typename D = derived_type>
            inline auto initialize_k() const
              -> std::enable_if_t<!decltype(meta::sig_init_k_f(std::declval<D>()))::value, k_tensor_type>
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    return std::make_tuple(xt::xtensor<value_type, 2>{}, xt::xtensor<value_type, 2>{});
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    return k_tensor_type{};
                }
                else
                {
                    throw std::runtime_error("Missing initialize_k function!");
                }
            }
            template<typename D = derived_type>
            inline auto initialize_k() const
              -> std::enable_if_t<decltype(meta::sig_init_k_f(std::declval<D>()))::value, k_tensor_type>
            {
                return this->derived_cast().initialize_k_impl();
            }

            template<typename Cell>
            auto apply_multipoles_preprocessing(Cell& current_cell, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                decltype(meta::sig_preprocess_f(std::declval<derived_type>(), current_cell, thread_id))::value, void>
            {
                return this->derived_cast().apply_multipoles_preprocessing_impl(current_cell, thread_id);
            }
            // default fallback
            template<typename Cell>
            auto apply_multipoles_preprocessing(Cell& current_cell, [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<
                !decltype(meta::sig_preprocess_f(std::declval<derived_type>(), current_cell, thread_id))::value, void>
            {
            }

            template<typename Cell>
            auto apply_multipoles_postprocessing(Cell& current_cell, [[maybe_unused]] buffer_type const& products,
                                                 [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<decltype(meta::sig_postprocess_f(std::declval<derived_type>(), current_cell, products,
                                                                   thread_id))::value,
                                  void>
            {
                return this->derived_cast().apply_multipoles_postprocessing_impl(current_cell, products, thread_id);
            }

            template<typename Cell>
            auto apply_multipoles_postprocessing(Cell& current_cell, [[maybe_unused]] buffer_type const& products,
                                                 [[maybe_unused]] size_type thread_id = 0) const
              -> std::enable_if_t<!decltype(meta::sig_postprocess_f(std::declval<derived_type>(), current_cell,
                                                                    products, thread_id))::value,
                                  void>
            {
            }
            /**
             * @brief  Compute the matrix vector product for the M2L operator
             *
             * The operation is
             *    locals := scale_factor*A*multipoles + beta*locals,   or   y := alpha*A**T*x + beta*y,
             *  if the settings is options::dense_ classical matrix vector product
             *  and if the settings is options::low_rank_ A = UV we perform two matrix vector product
             *
             * @tparam T
             * @tparam T2
             * @param multipoles the multipole values
             * @param locals  the local values
             * @param tmp a working array need for low rank approximation
             * @param knm the kernel matrix
             * @param scale_factor the scaling factor
             * @param acc if true we accumulate (beta=1.0 0 otherwise)
             */
            template<typename T, typename T2>
            inline auto product(T const& multipoles, T& locals, T2& tmp, k_tensor_type const& knm,
                                value_type scale_factor, bool acc) const -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    auto const& u = meta::get<0>(knm);
                    auto const& v = meta::get<1>(knm);
                    tensor::blas2_product(multipoles, tmp, v, value_type(1.0), false, true);
                    tensor::blas2_product(tmp, locals, u, scale_factor, acc, false);
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    tensor::blas2_product(multipoles, locals, knm, scale_factor, acc);
                }
                else
                {
                    throw std::runtime_error("No implementation found for m2l product !");
                }
            }
            template<typename T>
            inline auto product(T const& multipoles, T& locals, k_tensor_type const& knm, value_type scale_factor,
                                bool acc) const -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    auto const& u = meta::get<0>(knm);
                    auto const& v = meta::get<1>(knm);
                    T tmp(locals.shape(), value_type(0.));
                    tensor::blas2_product(multipoles, tmp, v, value_type(1.0), false, true);
                    tensor::blas2_product(tmp, locals, u, scale_factor, acc, false);
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    tensor::blas2_product(multipoles, locals, knm, scale_factor, acc);
                }
                else
                {
                    throw std::runtime_error("No implementation found for m2l product !");
                }
            }

            /**
             * @brief  Compute the matrix vector product for the M2L operator
             *
             * The operation is
             *    locals := scale_factor*A*multipoles + beta*locals,   or   y := alpha*A**T*x + beta*y,
             *  if the settings is options::dense_ classical matrix vector product
             *  and if the settings is options::low_rank_ A = UV we perform two matrix vector product
             *
             * @tparam T
             * @tparam T2
             * @param multipoles the multipole values
             * @param locals  the local values
             * @param tmp a working array need for low rank approximation
             * @param knm the kernel matrix
             * @param tmp A temporary matrix used in low rank approximation
             * @param nb_mult the number of multipole to treat (number of column of multipoles)
             * @param scale_factor the scaling factor
             * @param acc if true we accumulate (beta=1.0 0 otherwise)
             */
            template<typename T, typename T1>
            inline auto product_m(T const& multipoles, T& locals, k_tensor_type const& Knm, T1& tmp, int nb_mult,
                                  value_type scale_factor, bool acc) const -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    //  Knm = U V^T
                    auto const& U = meta::get<0>(Knm);
                    auto const& V = meta::get<1>(Knm);
                    constexpr bool accumulate = false;

                    constexpr bool transposeV = true;

                    tensor::blas3_product(multipoles, tmp, V, nb_mult, value_type(1.0), accumulate, transposeV);

                    tensor::blas3_product(tmp, locals, U, nb_mult, scale_factor, acc);
                }
                else if constexpr(std::is_same_v<settings, options::dense_>)
                {
                    tensor::blas3_product(multipoles, locals, Knm, nb_mult, scale_factor, acc);
                }
                else
                {
                    throw std::runtime_error("No implementation found for m2l product !");
                }
            }

            template<typename Cell>
            auto apply_m2l_single(Cell const& source_cell, Cell& target_cell, std::size_t neighbor_idx,
                                  std::size_t tree_level, [[maybe_unused]] buffer_type& products,
                                  [[maybe_unused]] size_type thread_id = 0) const -> void
            {
                std::size_t level{};
                scale_factor_type scale_factor{};

                if constexpr(homogeneity_tag == matrix_kernels::homogeneity::homogenous)
                {
                    level = 0;
                    // here we scale the target cell width to (homogenous case) match
                    // the [-1,1] unitary cell used to generate the corresponding
                    // interaction matrix.
                    scale_factor = m_far_field.scale_factor((target_cell.width()) / value_type(2.));
                }
                else   // non-homogenous case.
                {
                    // root level is 0 (i.e the simulation box) first level of cell is level 1
                    // we start the indexing of matrixes at 0, hence the -1 on the cell level
                    level = tree_level - 2;
                    scale_factor.fill(1.0);
                }
                if constexpr(symmetry_support)
                {
                    // the kernel is symmetric
                    // Decrease the number of matrix-vector product due to the symmetry
                    // see https://hal.inria.fr/hal-00746089v2
                    const auto neighbor_sym = this->symmetry_k_index(neighbor_idx);
                    const auto number_of_interactions = number_of_matrices_in_orthant<dimension>();

                    auto const& k = m_interactions_matrices.at(level * number_of_interactions + neighbor_sym);
                    auto& permuted_multipoles = products.at(0);
                    auto& permuted_locals = products.at(1);

                    for(std::size_t n = 0; n < kn; ++n)
                    {
                        for(std::size_t m = 0; m < km; ++m)
                        {
                            auto const& multipoles = source_cell.cmultipoles(m);
                            auto& locals = target_cell.locals(n);
                            // the multipole is permuted
                            const auto m_ptr = multipoles.data();
                            const auto l_ptr = locals.data();
                            const auto m_p_ptr = permuted_multipoles.data();
                            const auto l_p_ptr = permuted_locals.data();
                            const auto perm_ptr = &m_sym_permutations.at(neighbor_idx, 0);

                            for(std::size_t i{0}; i < m_nnodes; ++i)
                            {
                                m_p_ptr[perm_ptr[i]] = m_ptr[i];
                            }

                            this->product(permuted_multipoles, permuted_locals, k.at(n, m), scale_factor.at(n), false);

                            // the local expansion is permuted to the original.
                            for(std::size_t i{0}; i < m_nnodes; ++i)
                            {
                                l_ptr[i] += l_p_ptr[perm_ptr[i]];
                            }
                        }
                    }
                }
                else if constexpr(std::is_same_v<settings, options::low_rank_> ||
                                  std::is_same_v<settings, options::dense_>)
                {
                    // non symmetric kernel and low rank or dense approximation of the kernel

                    auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                    auto const& multipoles = source_cell.cmultipoles();
                    auto& locals = target_cell.locals();   // we generate km*kn products
                    // loop on km
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        // meta loop on kn
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            this->product(multipoles.at(m), locals.at(n), k.at(n, m), scale_factor.at(n), true);
                        }
                    }
                }
                else
                {
                    // non symmetric kernel and specific product (fft for uniform approximation)
                    //                    auto const& multipoles = source_cell.cmultipoles();
                    //                    auto& locals = target_cell.locals();
                    auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                    // we generate km*kn products
                    // loop on km
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        // meta loop on kn
                        for(std::size_t n = 0; n < kn; ++n)
                        {
                            this->derived_cast().apply_m2l_impl(source_cell, target_cell, products, k, scale_factor, n,
                                                                m, thread_id);
                        }
                    }
                }
            }
            template<typename Cell>
            auto apply_m2l_loop(Cell& target_cell, std::size_t tree_level, [[maybe_unused]] buffer_type& products,
                                [[maybe_unused]] size_type thread_id = 0) const
            {
                std::size_t level{};
                scale_factor_type scale_factor{};
                //
                if constexpr(homogeneity_tag == matrix_kernels::homogeneity::homogenous)
                {
                    level = 0;
                    // here we scale the target cell width to (homogenous case) match
                    // the [-1,1] unitary cell used to generate the corresponding
                    // interaction matrix.
                    scale_factor = m_far_field.scale_factor((target_cell.width()) / value_type(2.));
                }
                else   // non-homogenous case.
                {
                    // root level is 0 (i.e the simulation box) first level of cell is level 1
                    // we start the indexing of matrixes at 0, hence the -1 on the cell level
                    level = tree_level - 2;
                    scale_factor.fill(1.0);
                }
                auto const& cell_symbolics = target_cell.csymbolics();
                auto const& interaction_positions = cell_symbolics.interaction_positions;
                auto const& interaction_iterators = cell_symbolics.interaction_iterators;
                // #ifdef M2L_NEW_SYM
                // for symmetry support inner_type of the buffer for low_rank approximation otherwise the type of the
                // local approximation
                using local_type = std::conditional_t<std::is_same_v<settings, options::low_rank_>,
                                                      std::conditional_t<symmetry_support, buffer_inner_type,
                                                                         std::decay_t<decltype(target_cell.locals(0))>>,
                                                      meta::empty>;
                // #else
                //                 using local_type = std::decay_t<decltype(target_cell.locals(0))>;
                // #endif
                local_type work;

                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    // //
                    if constexpr(symmetry_support)
                    {
                        // rank x max_symm
                        work.resize(products.at(1).shape());
                        // work.resize(std::array<std::size_t, 2>{m_nnodes, largest_number_permutation<dimension>()});
                    }
                    else
                    {
                        work.resize(target_cell.locals(0).shape());
                    }
                }
                //
                // Selection product operator depending on symmetry, product optimization
                //
                if constexpr(symmetry_support)
                {
                    // the kernel is symmetric
                    // Decrease the number of matrix-vector product due to the symmetry
                    // see https://hal.inria.fr/hal-00746089v2

                    constexpr auto number_of_symmetries = number_of_matrices_in_orthant<dimension>();

                    std::array<std::array<int, largest_number_permutation<dimension>()>, number_of_symmetries>
                      neighbors_perm;
                    std::array<int, number_of_symmetries> number_of_permutation{};
                    //
                    // access the buffer to store temporary array
                    //  aggregate_multipoles and  aggregate_locals are vectors of
                    //  xt::xarray of size number of permutation
                    auto aggregate_multipoles = products.at(0);
                    auto aggregate_locals = products.at(1);
                    // set cells in neighbors_perm according to their symmetry
                    for(std::size_t index{0}; index < cell_symbolics.existing_neighbors; ++index)
                    {
                        auto const& neighbor_idx = interaction_positions.at(index);
                        const auto neighbor_sym = this->symmetry_k_index(neighbor_idx);
                        //
                        neighbors_perm[neighbor_sym][number_of_permutation[neighbor_sym]] = index;
                        number_of_permutation[neighbor_sym] += 1;
                    }

                    // Loop on the number of rhs (input)
                    for(std::size_t m = 0; m < km; ++m)
                    {
                        // loop on the number of symmetries
                        for(std::size_t idx{0}; idx < number_of_symmetries; ++idx)
                        {
                            if(number_of_permutation[idx] > 0)
                            {
                                auto const& neighs = neighbors_perm[idx];
                                // Fill the temporary array with all multipoles
                                for(int i = 0; i < number_of_permutation[idx]; ++i)
                                {
                                    auto const& index = neighs[i];
                                    auto const& neighbor_idx = interaction_positions.at(index);
                                    // get the permutation associated to the position of the cell
                                    const auto perm_ptr = &m_sym_permutations.at(neighbor_idx, 0);
                                    //
                                    // Gather the multipoles for the permutation
                                    // get the set of multipole for the current symmetry
                                    // get the xt::xarray of aggregated multipole for the current symmetry
                                    // permuted_multipoles is a xarray of size(max cells in symmetry, the grid of
                                    // nodes)
                                    //   the order of the tensor is dimension + 1
                                    // get multipole
                                    auto const& source_cell = *interaction_iterators.at(index);
                                    auto const& multipoles = source_cell.cmultipoles(m);
                                    // get the pointer of the source multipole
                                    const auto m_ptr = multipoles.data();

                                    for(std::size_t j{0}; j < m_nnodes; ++j)
                                    {
                                        aggregate_multipoles(perm_ptr[j], i) = m_ptr[j];
                                        // current_permuted_multipole_ptr[perm_ptr[j]] = m_ptr[j];
                                    }
                                }   // end of the aggregation for the current symmetry idx

                                //////////////////////////////////////////////////////////////
                                // Performed the matrix matrix  product
                                //
                                auto const& k = m_interactions_matrices.at(level * number_of_symmetries + idx);

                                for(std::size_t n = 0; n < kn; ++n)
                                {
                                    auto& locals = target_cell.locals(n);
                                    const auto l_ptr = locals.data();
                                    // Perform matrix-matrix product with aggregate multipoles
                                    this->product_m(aggregate_multipoles, aggregate_locals, k.at(n, m), work,
                                                    number_of_permutation[idx], scale_factor.at(n), false);
                                    //

                                    // the local expansion is permuted to their original.
                                    for(int i = 0; i < number_of_permutation[idx]; ++i)
                                    {
                                        auto const& neighbor_idx = interaction_positions.at(neighs[i]);
                                        // get the permutation associated to the position of the cell
                                        const auto perm_ptr = &m_sym_permutations.at(neighbor_idx, 0);
                                        // add current multipole  to aggregate_multipoles
                                        // auto current_permuted_locals = aggregate_locals.data() + m_nnodes * i;
                                        for(std::size_t j{0}; j < m_nnodes; ++j)
                                        {
                                            l_ptr[j] += aggregate_locals(perm_ptr[j], i);
                                        }
                                    }

                                }   // end kn loop on the output of the matrix kernel
                            }       // end if
                        }           // end loop number of cells in the symmetry
                    }               // end km loop on the input of the matrix kernel

                }   // end constexpr
                else if constexpr(std::is_same_v<settings, options::low_rank_> ||
                                  std::is_same_v<settings, options::dense_>)
                {
                    // non symmetric kernel and low rank or dense approximation of the kernel

                    for(std::size_t index{0}; index < cell_symbolics.existing_neighbors; ++index)
                    {
                        auto const& source_cell = *interaction_iterators.at(index);
                        const auto neighbor_idx = static_cast<std::size_t>(interaction_positions.at(index));

                        auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                        // we generate km*kn products
                        // loop on km
                        auto const& multipoles = source_cell.cmultipoles();
                        auto& locals = target_cell.locals();

                        for(std::size_t m = 0; m < km; ++m)
                        {
                            // meta loop on kn
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                this->product(multipoles.at(m), locals.at(n), work, k.at(n, m), scale_factor.at(n),
                                              true);
                            }
                        }
                    }
                }
                else
                {
                    // non symmetric kernel and specific product (fft for uniform approximation)

                    for(std::size_t index{0}; index < cell_symbolics.existing_neighbors; ++index)
                    {
                        auto const& source_cell = *interaction_iterators.at(index);
                        const auto neighbor_idx = static_cast<std::size_t>(interaction_positions.at(index));

                        auto const& k = m_interactions_matrices.at(level * m_m2l_interactions + neighbor_idx);
                        // we generate km*kn products
                        // loop on km
                        for(std::size_t m = 0; m < km; ++m)
                        {
                            // meta loop on kn
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                this->derived_cast().apply_m2l_impl(source_cell, target_cell, products, k, scale_factor,
                                                                    n, m, thread_id);
                            }
                        }
                    }
                }
            }

          private:
            inline auto generate_interactions_matrices(size_type order, value_type width, std::size_t tree_height)
              -> void
            {
                std::size_t number_of_level{1};
                std::size_t number_of_interactions{0};
                value_type local_cell_width_extension{0};

                if constexpr(symmetry_support)
                {
                    number_of_interactions = number_of_matrices_in_orthant<dimension>();
                }
                else
                {
                    number_of_interactions = this->m2l_interactions();
                }

                // here width is the root width
                // so first level of cell is of width because we skip level 0 (the root) and (the first level)):
                value_type current_width{width};
                const value_type half{0.5};
                const value_type quarter{0.25};
                // we get the half width to scale the roots

                if constexpr(homogeneity_tag == matrix_kernels::homogeneity::non_homogenous)
                {
                    // (tree_heigh = 4 -> [0-3]) the bottom cells and leaves have the same symbolic level !
                    // but we remove level 0 (the root ie simulation box) and level 1.
                    number_of_level = tree_height - 2;
                    current_width = width * quarter;
                    local_cell_width_extension = m_cell_width_extension;
                }

                value_type half_width{current_width * half};
                // we need to keep the tensorial view
                // let the same view goes down to tensorial of point
                // X and Y are Td tensors storing point.

                // get the ref of the vector
                auto& interactions_matrices{this->interactions_matrices()};

                // resizing and initializing the vector
                // homogenous -> only one level
                // non_homogenous -> we generate interaction matrices for each tree level processed by the m2l
                // operator
                interactions_matrices.resize(
                  number_of_interactions * number_of_level,
                  // xtensor_fixed<xshape<km,kn>>
                  interaction_matrix_type(typename interaction_matrix_type::shape_type{},
                                          // in Chebyshev, K_mn is a matrix of size [nnodes, nnodes]
                                          initialize_k(), xt::layout_type::row_major));
                // loop on levels
                for(std::size_t l{0}; l < number_of_level; ++l)
                {
                    // homogenous -> X is the [-1,1] box reference
                    // non_homogenous -> X is the [-cell_width_at_level/2, cell_width_at_level/2]
                    // X is a multidimensional grid generator returning a grid for X, another one for Y,...
                    // X is a multidimensional grid of points scaled on the size of cell
                    auto X_points = tensor::generate_grid_of_points<dimension>(
                      (half_width + half * local_cell_width_extension) * m_roots);

                    // lambda for generation the matrixes corresponding to one interaction
                    std::size_t flat_idx{0};
                    auto generate_all_interactions = [order, this, &interactions_matrices, &flat_idx, &X_points,
                                                      current_width, number_of_interactions, l](auto... is)
                    {
                        if(((std::abs(is) > separation_criterion) || ...))
                        {
                            // constructing centers to generate Y
                            xt::xarray<container::point<value_type, dimension>> centers(std::vector(dimension, order));
                            xt::xarray<container::point<value_type, dimension>> Y_points(std::vector(dimension, order));

                            // here we fill the centers with the current loop indexes
                            // X_points is scaled with the width of cell so, Y_points will be scaled directly
                            centers.fill(
                              container::point<value_type, dimension>({(value_type(is) * current_width)...}));
                            // then we calculate the Y points
                            // TODO : add directly the point value_type here.
                            Y_points = X_points + centers;

                            // we get the ref of interaction matrices to generate
                            auto& nm_fc_tensor = interactions_matrices.at(l * number_of_interactions + flat_idx);
                            // and we generate each matrix needed for the product (kn*km matrices)
                            for(std::size_t n = 0; n < kn; ++n)
                            {
                                for(std::size_t m = 0; m < km; ++m)
                                {
                                    nm_fc_tensor.at(n, m) = std::move(generate_matrix_k(X_points, Y_points, n, m));
                                }
                            }
                        }
                        ++flat_idx;
                    };

                    if constexpr(symmetry_support)
                    {
                        // we generate only the matrices in the positive cone of symmetries.
                        meta::looper_symmetries<dimension>{}(generate_all_interactions);
                    }
                    else
                    {
                        // loop range [-3,4[, ie range concept exclude the last value.
                        std::array<int, dimension> starts{};
                        std::array<int, dimension> stops{};
                        starts.fill(-3);
                        stops.fill(4);
                        // here we expand at compile time d loops of the range
                        // the indices of the d loops are input parameters of the lambda generate_all_interactions
                        meta::looper_range<dimension>{}(generate_all_interactions, starts, stops);
                    }

                    // we divide the widths for the next tree level
                    current_width *= half;
                    half_width *= half;
                }
            }

          protected:
            /// @brief Initialise the interaction matrices K.
            ///
            /// Homogenous : we construct only K on the points in [-1,1]^d cell, the length is
            /// then 2. The matrix is applied on a cell of size width, then we have to scale
            /// by cell_width/2.
            /// Non-homogenous : The interaction matrices are constructed at the from root level
            /// to the bottom of the tree (last cell level ie. same as the leaf level)
            /// hence, we don't need the scale_factor of the kernel.
            /// @param order : number of therms of the polynomial approximation. order^d is the number of grid points.
            /// @param root_cell_width : width of the top root cell needed for the non-homogenous case.
            /// @param tree_height : hight of the tree.
            ///
            /// @return
            inline auto initialize(size_type order, value_type root_cell_width, std::size_t tree_height) -> void
            {
                if constexpr(std::is_same_v<settings, options::low_rank_>)
                {
                    m_weights_of_roots = generate_weights(m_order);
                }
                generate_interactions_matrices(
                  order,
                  (homogeneity_tag == matrix_kernels::homogeneity::non_homogenous) ? root_cell_width : value_type(2.),
                  tree_height);
            }

            template<typename D = derived_type>
            [[nodiscard]] inline auto generate_weights(std::size_t order) const
              -> std::enable_if_t<decltype(meta::sig_gen_w_f(std::declval<D>(), order))::value, array_type>
            {
                return this->derived_cast().generate_weights_impl(order);
            }
            template<typename D = derived_type>
            [[nodiscard]] inline auto generate_weights(std::size_t order) const
              -> std::enable_if_t<!decltype(meta::sig_gen_w_f(std::declval<D>(), order))::value, array_type>
            {
                return array_type{};
            }

            [[nodiscard]] inline auto derived_cast() & noexcept -> derived_type&
            {
                return *static_cast<derived_type*>(this);
            }

            [[nodiscard]] inline auto derived_cast() const& noexcept -> derived_type const&
            {
                return *static_cast<const derived_type*>(this);
            }

            [[nodiscard]] inline auto derived_cast() && noexcept -> derived_type
            {
                return *static_cast<derived_type*>(this);
            }

          private:
            std::vector<interaction_matrix_type, XTENSOR_DEFAULT_ALLOCATOR(interaction_matrix_type)>
              m_interactions_matrices{};
            sym_permutations_type m_sym_permutations{};
            k_indices_type m_k_indices{};
            matrix_kernel_type m_far_field{};
            xt::xarray<value_type>
              m_weights_of_roots{};   ///< the weight associated to the roots of the m_order-1 polynomial
            const size_type m_m2l_interactions{};
            const size_type m_nnodes{};     ///< number of modes m_order^dimension
            const size_type m_order{};      ///< number of terms of the expansion (1d)
            const array_type m_roots{};     ///< the roots of the m_order-1 polynomial
            const value_type m_epsilon{};   ///< the accuracy for low-rank approximation (10^(-o)) @todo check 10^(1-o)
            const value_type m_cell_width_extension{};   // w width of the extension of the cell
        };
    }   // namespace impl
}   // namespace scalfmm::interpolation

#endif   // SCALFMM_INTERPOLATION_UNIFORM_HPP
