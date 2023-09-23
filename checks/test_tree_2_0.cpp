#include "parameters.hpp"
//#include "FCountKernel.hpp"

#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/tensor.hpp"

#include <Components/FSymbolicData.hpp>
#include <Files/FFmaGenericLoader.hpp>
#include <GroupTree/Core/FGroupSeqAlgorithm.hpp>
#include <GroupTree/Core/FGroupTree.hpp>
#include <GroupTree/Core/FP2PGroupParticleContainer.hpp>
#include <Kernels/Interpolation/FInterpMatrixKernel.hpp>
#include <Kernels/Uniform/FUnifCell.hpp>
#include <Kernels/Uniform/FUnifKernel.hpp>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    // Parameter handling
    args::cli.parse(argc, argv);
    command_line_parameters params(args::cli);

    // Getting command line parameters
    const auto tree_height{params.tree_height};
    const auto input_file{params.input_file};
    const auto group_size{params.block_size};

    static constexpr std::size_t dimension = 3;
    static constexpr std::size_t number_of_physical_values = 1;
    static constexpr auto number_of_attributes = dimension + number_of_physical_values;
    static constexpr int order = 2;
    // ---------------------------------------
    // scalfmm 2.0 tree tests and benchmarks.
    // ---------------------------------------
    using value_type = double;
    using matrix_kernel_type = FInterpMatrixKernelR<value_type>;
    // using cell_type = TestCountNodeData;
    using cell_type = FUnifCell<value_type, order>;
    using cell_up_type = typename cell_type::multipole_t;
    using cell_down_type = typename cell_type::local_expansion_t;
    using cell_symbolic_type = FSymbolicData;
    using container_particle_type = FP2PGroupParticleContainer<value_type>;
    using grouptree_type =
      FGroupTree<value_type, cell_symbolic_type, cell_up_type, cell_down_type, container_particle_type,
                 number_of_physical_values, number_of_attributes, value_type>;
    // using kernel_type = FCountKernel<cell_type, FP2PParticleContainer<value_type>, container_particle_type>;
    using kernel_type = FUnifKernel<value_type, cell_type, container_particle_type, matrix_kernel_type, order>;
    using algo_type =
      FGroupSeqAlgorithm<grouptree_type, typename grouptree_type::CellGroupClass, cell_type, kernel_type,
                         typename grouptree_type::ParticleGroupClass, container_particle_type>;

    // Open particle file
    double box_width{1.};
    FPoint<double> box_center{0., 0., 0.};

    double step{box_width / scalfmm::math::pow(2, (tree_height - 1))};
    std::cout << "Step = " << step << '\n';

    auto number_of_values_per_dimension = std::size_t(scalfmm::math::pow(2, (tree_height - 1)));
    std::cout << "Number of value per dimension = " << number_of_values_per_dimension << '\n';

    xt::xarray<std::tuple<double, double, double, double>> particles(
      std::vector(dimension, number_of_values_per_dimension));

    std::cout << "linspace = "
              << xt::linspace(double(-box_width / 2.) + step * 0.5, double(box_width / 2.) - step * 0.5,
                              number_of_values_per_dimension)
              << '\n';

    auto particle_generator = scalfmm::tensor::generate_meshgrid<dimension>(xt::linspace(
      double(-box_width / 2.) + step * 0.5, double(box_width / 2.) - step * 0.5, number_of_values_per_dimension));

    auto eval_generator = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::eval(std::forward<decltype(xs)>(xs))...); }, particle_generator);
    auto flatten_views = std::apply(
      [](auto&&... xs) { return std::make_tuple(xt::flatten(std::forward<decltype(xs)>(xs))...); }, eval_generator);

    auto particle_flatten_views = xt::flatten(particles);

    FP2PParticleContainer<value_type> all_particles{};

    // Filling container from file
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        auto part = std::apply([&i](auto&&... xs) { return FPoint<value_type>{std::forward<decltype(xs)>(xs)[i]...}; },
                               flatten_views);
        all_particles.push(part, i);
    }

    // Put the data into the tree
    grouptree_type gtree(tree_height, box_width, box_center, group_size, &all_particles);
    gtree.printInfoBlocks();

    // Run the algorithm
    const matrix_kernel_type mk{};
    kernel_type unif_kernel(tree_height, box_width, box_center, &mk);
    // kernel_type count_kernel{};
    algo_type galgo(&gtree, &unif_kernel);
    galgo.execute();

    return 0;
}
