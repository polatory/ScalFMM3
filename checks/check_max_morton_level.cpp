/*
 * genarateDistributions.cpp
 *
 *  Created on: 23 mars 2014
 *      Author: Olivier Coulaud
 */

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
//
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/utils/parameters.hpp"
#include "scalfmm/utils/sort.hpp"

#include <cpp_tools/cl_parser/cl_parser.hpp>
#include <cpp_tools/colors/colorized.hpp>

//
/// \file  compare_files.cpp
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
//!            first file name to compare (with extension .fma (ascii) or bfma (binary)</li>
//!
//!      <li>    --input-file2, -fin2 std::string
//!             second file name to compare (with extension .fma (ascii) or bfma (binary) </li>
//!
//!     <li> --sort
//!      Sort particles according to their morton index. </li>
//!
//!     <li> --index, --index1 value,value,value,...  (required)
//!       column index for data associated to particle (x,y,z,input data, output data)
//!        specify the list of index to evaluate the error </li>
//!
//!    <li>--index2 value,value,value,... (if not specified index1 is used)
//!       column index for data associated to particle (x,y,z,input data, output data)
//!       specify the list of index to evaluate the error </li>
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

using value_type = double;

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
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct input_file_two
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file2", "-fin2"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;               //
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct sort_particle
    {
        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--sort"};
        std::string description = "Sort the particles to their morton index";
    };
    struct index_to_compare : cpp_tools::cl_parser::required_tag
    {
        std::vector<std::string> flags = {"--index1"}; /*!< The flags */
        std::string description = "For data associated to particle (x,y,z,input data, output data)\n  "
                                  "specify the list of index to evaluate the error"; /*!< The description */
        std::string input_hint = "value,value,value,...";                            /*!< The description */

        /**\brief Type definition
         *
         * We define directly the type into the parameter and it inherits of
         * std::vector for simplicity.
         */
        struct type : std::vector<int>
        {
            /// Operator >>
            friend std::istream& operator>>(std::istream& is, type& t)
            {
                bool first = true;
                int val;
                char delimiter;
                while(!is.eof())
                {
                    // The first object is not delimited
                    first ? (first = false) : ((is >> delimiter), true);
                    // Get the value
                    is >> val;
                    t.emplace_back(val);
                }
                return is;
            }
        };
    };
    struct index2_to_compare
    {
        std::vector<std::string> flags = {"--index2"}; /*!< The flags */
        std::string description = "For data associated to particle (x,y,z,input data, output data)\n  "
                                  "specify the list of index to evaluate the error"; /*!< The description */
        std::string input_hint = "value,value,value,...";                            /*!< The description */

        /**\brief Type definition
         *
         * We define directly the type into the parameter and it inherits of
         * std::vector for simplicity.
         */
        struct type : std::vector<int>
        {
            /// Operator >>
            friend std::istream& operator>>(std::istream& is, type& t)
            {
                bool first = true;
                int val;
                char delimiter;
                while(!is.eof())
                {
                    // The first object is not delimited
                    first ? (first = false) : ((is >> delimiter), true);
                    // Get the value
                    is >> val;
                    t.emplace_back(val);
                }
                return is;
            }
        };
    };
}   // namespace local_args
///////////////////////////////////////////////////////////**
template<int Dimension>   //, typename VectorType>
auto cmp(std::string const& file1) -> int
{
    using position_type = scalfmm::container::point<value_type, Dimension>;
    using box_type = scalfmm::component::box<position_type>;
    bool verbose{true}, sort_particle{true};

    scalfmm::io::FFmaGenericLoader<value_type, Dimension> loader1(file1, verbose);

    //
    // Allocation
    //
    const auto nb_particles = loader1.getNumberOfParticles();
    const auto nb_data_per_particle1 = loader1.getNbRecordPerline();

    //    if ((nb_data_per_particle != loader2.getNbRecordPerline())) {
    //      std::cerr << "Wrong files only "
    //                << std::min(loader2.getNbRecordPerline(), nb_data_per_particle)
    //                << " to read." << std::endl;
    //      std::exit(EXIT_FAILURE);
    //    }
    std::vector<value_type> particles1(nb_particles * nb_data_per_particle1);
    //
    loader1.fillParticles(particles1, nb_particles);

    if(sort_particle)
    {
        // define a box, used in the sort
        using morton_type = std::size_t;
        const std::size_t max_level = (sizeof(morton_type) * 8 / Dimension) - 1;
        //
        box_type box(loader1.getBoxWidth(), loader1.getCenterOfBox());

        std::cout << "Sort needed !! " << std::endl;
        scalfmm::utils::sort_raw_array_with_morton_index(box, nb_particles, particles1);
    }
    return 0;
}
//
//////////////////////////////////////////////////////////////////////////////
//
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    /// Parsing options
    ///
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::Dimension{}, local_args::input_file_one());

    // Parameter handling
    parser.parse(argc, argv);

    std::cout << cpp_tools::colors::blue << "Entering sort_particles...\n" << cpp_tools::colors::reset;
    const int dimension{parser.get<args::Dimension>()};
    const auto filename1{parser.get<local_args::input_file_one>()};
    if(!filename1.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file 1: " << filename1 << cpp_tools::colors::reset
                  << '\n';
    }

    // scalfmm::meta::td<decltype(ind ex)> u1;

    if(dimension == 1)
    {
        return cmp<1>(filename1);
    }
    else if(dimension == 2)
    {
        return cmp<2>(filename1);
    }
    else if(dimension == 3)
    {
        return cmp<3>(filename1);
    }
    else
    {
        throw std::invalid_argument("The dimension is wrong (1,2 or 3)");
        return 1;
    }
}
