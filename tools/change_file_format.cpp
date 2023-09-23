/*
 * change_file_format.cpp
 *
 *  Created on: 23 mars 2014
 *      Author: Olivier Coulaud
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
//
//
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/vtk_writer.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/utils/compare_results.hpp"
#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>

//
/// \file  change_file_format.cpp
//!
//! \brief change_file_format: Driver to transform a FMA format (binary <-> Ascii)
//!
//!  Driver to transform a FMA format a<br>
//! For a description of the FMA format see FFmaGenericLoader<br>
//!  <b> General arguments:</b>
//!     \param   -help (-h)      to see the parameters available in this driver
//!     \param   -fin name:  file name  to convert (with extension .fma (ascii) or bfma (binary)
//!     \param   --old-fin:      to read the input file in scalFMM 2 format
//!     \param   --old-fout:      to write the output file in scalFMM 2 format
//!     \param   -fout name: file name for the transformed file  (.fma or .bfm)
//!     \param   --visu-file name: file name for the vtp file (Paraview format)
//!     \param   --dimension d:  The dimension of the space where the particles leave
//!
//! \b examples
//!  Transform an ascii file in a binary file<br>
//!   change_file_format -fin unitCubeXYZQ100.fma  -fout unitCubeXYZQ100.bfma

using value_type = double;
namespace local_args
{
    struct input_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };
    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
    };
    struct visu_file
    {
        using type = std::string;
        cpp_tools::cl_parser::str_vec flags = {"--visu-file"};
        std::string description = "Input filename (.vtp).";
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct dimensionSpace
    {
        cpp_tools::cl_parser::str_vec flags = {"--dim", "--dimension", "-d"};
        std::string description = "Dimension of the space (d < 5)";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
        type def = 3;
    };
    struct oldFinFormat
    {
        using type = bool;
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--old-fin"};
        std::string description = "Old format  for input(ScalFMM 1.5 and 2.0) for binary or ascii fma file";
    };
    struct oldFoutFormat
    {
        using type = bool;
        enum
        {
            flagged
        };
        cpp_tools::cl_parser::str_vec flags = {"--old-fout"};
        std::string description = "Old format  for output (ScalFMM 1.5 and 2.0) for binary or ascii fma file";
    };
}   // namespace local_args
template<int Dimension>
void change_format(const std::string& input_file, int& use_old_format, const std::string& visu_file,
                   const std::string& output_file)
{
    bool verbose = true;
    scalfmm::io::FFmaGenericLoader<value_type, Dimension> loader(input_file, verbose, (use_old_format == 1));

    const auto NbPoints = loader.getNumberOfParticles();
    const auto nbData = loader.getNbRecordPerline();
    const auto arraySize = nbData * NbPoints;

    std::vector<value_type> particles(arraySize, 0.0);

    std::size_t j = 0;
    for(std::size_t idxPart = 0; idxPart < NbPoints; ++idxPart, j += nbData)
    {
        loader.fillParticle(&particles[j], nbData);
    }

    /////////////////////////////////////////////////////////////////////////
    //                                           Save data
    /////////////////////////////////////////////////////////////////////////
    //
    //  Generate file for ScalFMM FMAGenericLoader
    //
    if(!output_file.empty())
    {
        scalfmm::io::FFmaGenericWriter<value_type> writer(output_file);
        auto nb_input = loader.get_number_of_input_per_record();
        auto dim = loader.get_dimension();
        if(use_old_format == 2)
        {
            writer.writeHeaderOld(loader.getCenterOfBox(), loader.getBoxWidth(), NbPoints, sizeof(value_type), nbData,
                                  dim, nb_input);
        }
        else
        {
            writer.writeHeader(loader.getCenterOfBox(), loader.getBoxWidth(), NbPoints, sizeof(value_type), nbData, dim,
                               nb_input);
        }
        writer.writeArrayOfReal(particles.data(), nbData, NbPoints);
    }
    if(!visu_file.empty())
    {
        std::string visufile(visu_file);
        scalfmm::tools::io::exportVTKxml(visufile, particles, Dimension, loader.get_number_of_input_per_record(),
                                         NbPoints);
    }
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    /// Parsing options
    ///
    auto parser = cpp_tools::cl_parser::make_parser(
      local_args::input_file{}, local_args::output_file{}, cpp_tools::cl_parser::help{}, local_args::dimensionSpace{},
      local_args::oldFinFormat{}, local_args::oldFoutFormat{}, local_args::visu_file{});

    // Parameter handling
    parser.parse(argc, argv);

    std::cout << cpp_tools::colors::blue << "Entering change_file_format...\n" << cpp_tools::colors::reset;

    const std::string input_file{parser.get<local_args::input_file>()};
    std::string output_file{""};
    std::string visu_file{""};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                  << '\n';
    }

    int use_old_format{0};
    if(parser.exists<local_args::oldFinFormat>())
    {
        use_old_format = 1;
    }
    else if(parser.exists<local_args::oldFoutFormat>())
    {
        use_old_format = 2;
    }
    auto dimension = parser.get<local_args::dimensionSpace>();
    if(parser.exists<local_args::output_file>())
    {
        output_file = parser.get<local_args::output_file>();
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    if(parser.exists<local_args::visu_file>())
    {
        if(dimension > 3)
        {
            std::cerr << "VTK export works only if dimension < 4" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        visu_file = parser.get<local_args::visu_file>();
        std::cout << cpp_tools::colors::blue << "<params> Visu file : " << visu_file << cpp_tools::colors::reset
                  << '\n';
    }
    std::cout << "use oldFormat: " << use_old_format << " true: " << true << std::endl;
    if(dimension == 1)
    {
        change_format<1>(input_file, use_old_format, visu_file, output_file);
    }
    else if(dimension == 2)
    {
        change_format<2>(input_file, use_old_format, visu_file, output_file);
    }
    else if(dimension == 3)
    {
        change_format<3>(input_file, use_old_format, visu_file, output_file);
    }
    else if(dimension == 4)
    {
        change_format<4>(input_file, use_old_format, visu_file, output_file);
    }
    else
    {
        std::cerr << "Dimension should be lower than 5 (d<5)" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return 0;
}
