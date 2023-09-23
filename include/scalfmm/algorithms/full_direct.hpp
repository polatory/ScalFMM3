// --------------------------------
// See LICENCE file at project root
// File : algorithm/full_direct.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_FULL_DIRECT_HPP
#define SCALFMM_ALGORITHMS_FULL_DIRECT_HPP

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
//

#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"
#include <scalfmm/container/iterator.hpp>
#include <scalfmm/container/point.hpp>
#include <scalfmm/matrix_kernels/laplace.hpp>
#include <scalfmm/utils/io_helpers.hpp>
#include <scalfmm/utils/static_assert_as_exception.hpp>

using namespace scalfmm::io;
namespace scalfmm::algorithms
{

    /**
     * @brief compute the direct particle interactions
     *
     * Compute the direct interactions between the particles inside the containers.
     *
     *   \f$ out(pt) = \sum_{ps \ne pt }{ matrix\_kernel(pt, ps) input(ps) }  \f$
     *
     * @param[inout] particles  the container of particles
     * @param[in] matrix_kernel the matrix kernel
     *
     */
    template<typename ContainerParticles, typename MatrixKernel>
    inline auto full_direct(ContainerParticles& particles, const MatrixKernel& matrix_kernel) -> void
    {
        //   using particle_type = Particle;
        using particle_type = typename ContainerParticles::value_type;
        using value_type = typename particle_type::position_type::value_type;
        // using range_outputs_type = typename particle_type::range_outputs_type;
        using matrix_type = typename MatrixKernel::template matrix_type<value_type>;

        static_assert(MatrixKernel::km == particle_type::inputs_size,
                      "Different input size between Matrix kernel and container!");
        static_assert(MatrixKernel::kn == particle_type::outputs_size,
                      "Different output size between Matrix kernel and container!");

        // #pragma omp parallel for shared(matrix_kernel)
        for(std::size_t idx = 0; idx < particles.size(); ++idx)
        {
            // Get proxy particle position
            auto pt_x = particles.at(idx).position();
            // Get proxy particle outputs
            auto val = particles.at(idx).outputs();

            // val the array of outputs
            matrix_type val_mat{};
            auto compute = [&pt_x, &val, &matrix_kernel, &val_mat, &particles](std::size_t start, std::size_t end)
            {
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    auto q = particles.at(idx_2).inputs();
                    auto pt_y = particles.at(idx_2).position();
                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < MatrixKernel::kn; ++j)
                    {
                        for(std::size_t i = 0; i < MatrixKernel::km; ++i)
                        {
                            val[j] += val_mat.at(j * MatrixKernel::km + i) * q[i];
                        }
                    }
                }
            };

            compute(0, idx);
            compute(idx + 1, particles.size());
        }
    }
    template<typename Particles, typename MatrixKernel>
    inline auto full_direct(std::vector<Particles>& particles, const MatrixKernel& matrix_kernel) -> void
    {
        //   using particle_type = Particle;
        using particle_type = Particles;
        using value_type = typename particle_type::position_type::value_type;
        // using range_outputs_type = typename particle_type::range_outputs_type;
        using matrix_type = typename MatrixKernel::template matrix_type<value_type>;

        static_assert(MatrixKernel::km == particle_type::inputs_size,
                      "Different input size between Matrix kernel and container!");
        static_assert(MatrixKernel::kn == particle_type::outputs_size,
                      "Different output size between Matrix kernel and container!");
        //       int idx{};

        // #pragma omp parallel for shared(matrix_kernel)
        for(std::size_t idx = 0; idx < particles.size(); ++idx)
        {
            // Get proxy particle position
            auto& pt_x = particles.at(idx).position();
            // Get proxy particle outputs
            auto& val = particles.at(idx).outputs();

            // val the array of outputs
            matrix_type val_mat{};
            auto compute = [&pt_x, &val, &matrix_kernel, &val_mat, &particles](std::size_t start, std::size_t end)
            {
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    auto q = particles.at(idx_2).inputs();
                    auto pt_y = particles.at(idx_2).position();
                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < MatrixKernel::kn; ++j)
                    {
                        for(std::size_t i = 0; i < MatrixKernel::km; ++i)
                        {
                            val[j] += val_mat.at(j * MatrixKernel::km + i) * q[i];
                        }
                    }
                }
            };

            compute(0, idx);
            compute(idx + 1, particles.size());
        }
    }
    /**
     * @brief compute the field due to the  particles the source container on the target particles
     *
     * compute the field due to the  particles the source container on the target particles by
     *
     *   \f$ out(pt) = \sum_{ps\in source} { matrix\_kernel(pt, ps) input(ps) }  \f$
     *
     * We don't check if | pt - ps| =0
     *
     * @param[in] particles_source  container of the source particles
     * @param[inout] particles_target  container of the target particles
     * @param[in] matrix_kernel  matrix kernel
     *
     */
    template<typename SourceParticles_type, typename TargetParticles_type, typename MatrixKernel>
    inline auto full_direct(std::vector<SourceParticles_type>& particles_source,
                            std::vector<TargetParticles_type>& particles_target, MatrixKernel const& matrix_kernel)
    {
        bool same_tree{false};
        if constexpr(std::is_same_v<TargetParticles_type, TargetParticles_type>)
        {
            same_tree = (&particles_source == &particles_target);
        }
        if(same_tree)
        {
            throw std::runtime_error("In p2p_outer the two containers must be different.");
        }
        //   using particle_type = Particle;
        using particle_source_type = SourceParticles_type;
        using particle_target_type = TargetParticles_type;
        using value_type = typename particle_source_type::position_type::value_type;
        // using range_outputs_type = typename particle_target_type::range_outputs_type;
        // using position_type = typename particle_source_type::position_type;
        using matrix_type = typename MatrixKernel::template matrix_type<value_type>;

        static_assert(MatrixKernel::km == particle_source_type::inputs_size,
                      "Different input size between Matrix kernel and container!");
        static_assert(MatrixKernel::kn == particle_target_type::outputs_size,
                      "Different output size between Matrix kernel and container!");
        //       int idx{};

        // #pragma omp parallel for simd shared(matrix_kernel)
        for(std::size_t idx = 0; idx < particles_target.size(); ++idx)
        {
            // const auto p = particles_target.particle(idx);
            auto& p = particles_target[idx];

            // // Get particle position
            auto const& pt_x = p.position();
            // // val is an alias on the array of outputs
            auto& val = p.outputs();
            matrix_type val_mat{};

            auto compute =
              [&pt_x, &val, &matrix_kernel, &val_mat, &particles_source](std::size_t start, std::size_t end)
            {
                // #pragma omp for simd
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    // auto ps = particles_source.particle(idx_2);
                    auto const& q = particles_source.at(idx_2).inputs();

                    auto const& pt_y = particles_source.at(idx_2).position();

                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < MatrixKernel::kn; ++j)
                    {
                        for(std::size_t i = 0; i < MatrixKernel::km; ++i)
                        {
                            val[j] += val_mat.at(j * MatrixKernel::km + i) * q[i];
                        }
                    }
                }
            };

            compute(0, particles_source.size());
        }
    }
    template<typename ContainerSourceParticles_type, typename ContainerTargetParticles_type, typename MatrixKernel>
    inline auto full_direct(ContainerSourceParticles_type& particles_source,
                            ContainerTargetParticles_type& particles_target, const MatrixKernel& matrix_kernel)
    {
        bool same_tree{false};
        if constexpr(std::is_same_v<ContainerSourceParticles_type, ContainerTargetParticles_type>)
        {
            same_tree = (&particles_source == &particles_target);
        }
        if(same_tree)
        {
            throw std::runtime_error("In p2p_outer the two containers must be different.");
        }
        //   using particle_type = Particle;
        using particle_source_type = typename ContainerSourceParticles_type::value_type;
        using particle_target_type = typename ContainerTargetParticles_type::value_type;
        using value_type = typename particle_source_type::position_type::value_type;
        // using range_outputs_type = typename particle_target_type::range_outputs_type;
        // using position_type = typename particle_source_type::position_type;
        using matrix_type = typename MatrixKernel::template matrix_type<value_type>;

        static_assert(MatrixKernel::km == particle_source_type::inputs_size,
                      "Different input size between Matrix kernel and container!");
        static_assert(MatrixKernel::kn == particle_target_type::outputs_size,
                      "Different output size between Matrix kernel and container!");
        //       int idx{};

        // #pragma omp parallel for simd shared(matrix_kernel)
        for(std::size_t idx = 0; idx < particles_target.size(); ++idx)
        {
            // Get proxy particle position
            auto pt_x = particles_target.at(idx).position();
            // Get proxy particle outputs
            auto val = particles_target.at(idx).outputs();
            matrix_type val_mat{};

            auto compute =
              [&pt_x, &val, &matrix_kernel, &val_mat, &particles_source](std::size_t start, std::size_t end)
            {
                // #pragma omp for simd
                for(std::size_t idx_2 = start; idx_2 < end; ++idx_2)
                {
                    // auto ps = particles_source.particle(idx_2);
                    auto q = particles_source.at(idx_2).inputs();

                    auto pt_y = particles_source.at(idx_2).position();

                    val_mat = matrix_kernel.evaluate(pt_x, pt_y);
                    for(std::size_t j = 0; j < MatrixKernel::kn; ++j)
                    {
                        for(std::size_t i = 0; i < MatrixKernel::km; ++i)
                        {
                            val[j] += val_mat.at(j * MatrixKernel::km + i) * q[i];
                        }
                    }
                }
            };

            compute(0, particles_source.size());
        }
    }

}   // namespace scalfmm::algorithms

#endif   // SCALFMM_ALGORITHMS_FULL_DIRECT_HPP
