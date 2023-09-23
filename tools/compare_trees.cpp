/*
 * compare_trees.cpp
 *
 *  Created on: 12 july 2023
 *      Author: Olivier Coulaud
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
//
#include "scalfmm/tools/tree_io.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/cell.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/io.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/compare_trees.hpp"

//
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"

#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>

//
/// \file  compare_trees.cpp
//!
//! \brief compare_files gives the error between two different files. The number of data per particles can be different
//!
//! Data should be in 3 d space otherwise we can change the value of the dimension.
//!  <b> General arguments:</b>
//! USAGE:
//! ./examples/RelWithDebInfo/compare_files [--help] [--input-file1 std::string] [--input-file2 std::string] [--sort
//! int] --index value,value,value,... [--index2 value,value,value,...]
//!
//!    DESCRIPTION:
//!< ul>
//!     <li>  --help, -h
//!            Display this help message </li>
//!
//!      <li>   --input-file1, -fin1 std::string
//!            first file name to compare (with extension .bin (binary)</li>
//!
//!      <li>    --input-file2, -fin2 std::string
//!             second file name to compare (with extension .bin (binary) </li>
//!
//! </ul>
//!
//! \b examples<p>
//!  \code
//!   Example 1) We compare the column 4 of the two files. We assume that the position of the points is the same
//!        compare2files -file1 unitCubeXYZQ100.fma  -file2 unitCubeXYZQ100.fma --index 4
//!
//!   Example 2)
//!   Here we compare data located on columns 5,6,7 for the date in file1 and columns 4,5,6 in file2. The two data
//!      are sorted by their morton index at level 20
//!
//!   compare_files -fin1 ref.fma  -fin2 res.fma --index 5,6,7 --index2 4,5,6 --sort
//!  \endcode

///
/////////////////////////////////////////////////////////////
///          Local parameters
/////////////////////////////////////////////////////////////
//
namespace local_args
{
    struct input_file_one
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file1", "-fin1"};
        std::string description = "Input binary filename 1(.";
        using type = std::string;
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct input_file_two
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file2", "-fin2"};
        std::string description = "Input binary filename 2.";
        using type = std::string;               //
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct comp
    {
      cpp_tools::cl_parser::str_vec flags = {"--compare", "-c"};
      std::string description = "1 multipoles, 2 locals, 3 both.";
      using type = int;               
      type def = 3;
      std::string input_hint = "std::string"; /*!< The input hint */
    };
}   // namespace local_args
/////////////////////////////////////////////////////////////

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    /// Parsing options
    ///
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, local_args::input_file_one(),
                                                    local_args::input_file_two(), local_args::comp());

    // Parameter handling
    parser.parse(argc, argv);

    const auto filename1{parser.get<local_args::input_file_one>()};
    if(!filename1.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file 1: " << filename1 << cpp_tools::colors::reset
                  << '\n';
    }

    const auto filename2{parser.get<local_args::input_file_two>()};
    if(!filename2.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file 2: " << filename2 << cpp_tools::colors::reset
                  << '\n';
    }
    const auto comp{parser.get<local_args::comp>()};

    //    bool verbose = true;
    //////////////////////////////////////////////////////////////////////////////////**
    //
    using value_type = double;
    constexpr int dimension = 2;
    //
    using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    constexpr int nb_inputs_near = matrix_kernel_type::km;
    constexpr int nb_outputs_near = matrix_kernel_type::kn;

    using interpolator_type =
      scalfmm::interpolation::interpolator<value_type, dimension, matrix_kernel_type,
                                           scalfmm::options::uniform_<scalfmm::options::low_rank_>>;

    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs_near, value_type, nb_outputs_near>;

    using position_type = typename particle_type::position_type;
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using box_type = scalfmm::component::box<position_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    auto tree1 = scalfmm::tools::io::read<group_tree_type>(filename1);
    scalfmm::io::trace(std::cout, tree1, 1);

    auto tree2 = scalfmm::tools::io::read<group_tree_type>(filename2);
    scalfmm::io::trace(std::cout, tree2, 1);

    value_type eps{1.e-8};

    if(scalfmm::utils::compare_two_trees(tree1, tree2, eps, comp))
    {
        std::cout << "Same trees !\n";
    }
    else
    {
        std::cout << "Trees are different!\n";
    }
    return 0;
}
