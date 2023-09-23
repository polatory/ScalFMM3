/*
 * sort_particles.cpp
 *
 *  Created on: 21 August 2020
 *      Author: Olivier Coulaud
 */
// @FUSE_OMP

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

//
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/compare_results.hpp"
#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>

/// \file  sort_particles.cpp
///
/// \brief sort_particles: Sort the particles according to their morton index
///
///  Driver to transform a FMA format and/or to build a visualization file<br>
/// For a description of the FMA format see FFmaGenericLoader<br>
///  <b> General arguments:</b>
///     \param   -help (-h)      to see the parameters available in this driver
///     \param   --dimension  dimension of teh space where the particles are
///     \param   --input_file  name1:  file name to read (with extension
///     .fma (ascii) or bfma (binary)
///     \param   --fout  name2: file nameto store the sorted particles comming from input_file
///     (with extension .fma (ascii) or bfma (binary)
/// sort_particles --input-file value [--output-file value] [--help]
///
/// DESCRIPTION:
///
///       --dimension, --d value
///            Dimension of the space
///
///     --input-file, -fin value
///            Input filename (.fma or .bfma).
///
///        --output-file, -fout value<p>
///     Output particle file (.fma or .bfma).
///
///     --help, -h
///    Display this help message
///
/// \b examples
/// \code
///   sort_particles --dim 3 --input_file unitCubeXYZQ100.fma  -fout unitCubeXYZQ100
/// \endcode

namespace loc_args
{

    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension of the space \n ";
        using type = int;
    };
    struct input_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };
    struct dataType
    {
        cpp_tools::cl_parser::str_vec flags = {"--use-float"};
        using type = bool;
        std::string description = "To generate float distribution";
        enum
        {
            flagged
        };
    };
    struct output_file : cpp_tools::cl_parser::required_tag

    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
    };
}   // namespace loc_args

template<int Dimension, typename value_type>
auto run(const std::string& input_file, const std::string& output_file, const bool verbose) -> int

{
    using position_type = scalfmm::container::point<value_type, Dimension>;
    using box_type = scalfmm::component::box<position_type>;

    scalfmm::io::FFmaGenericLoader<value_type, Dimension> loader(input_file, verbose);
    //
    auto nb_particles = loader.getNumberOfParticles();
    const unsigned int n_data_per_particle = loader.getNbRecordPerline();

    if(Dimension != loader.get_dimension())
    {
        std::cerr << "Wrong dimension in the call of sort_particles. The dimension should be " +
                       std::to_string(loader.get_dimension()) + "\n ";
        throw(" Wrong dimension in the call. \n");
    }

    std::vector<value_type> particles1(nb_particles * n_data_per_particle);
    //
    loader.fillParticles(particles1, nb_particles);
    //

    box_type box(loader.getBoxWidth(), loader.getCenterOfBox());

    scalfmm::utils::sort_raw_array_with_morton_index(box, nb_particles, particles1);

    std::cout << "Write outputs in " << output_file << std::endl;
    // const int level = 2;
    // for(int i = 0; i < nb_particles; ++i)
    // {
    //     position_type pos(&(particles1[i * n_data_per_particle]));
    //     std::cout << i << " p " << pos << " m " << scalfmm::index::get_morton_index(pos, box, level) << std::endl;
    // }

    scalfmm::io::FFmaGenericWriter<value_type> writer(output_file);
    writer.writeHeader(loader.getCenterOfBox(), loader.getBoxWidth(), nb_particles, sizeof(value_type),
                       n_data_per_particle, Dimension, loader.get_number_of_input_per_record());

    writer.writeArrayOfReal(particles1.data(), n_data_per_particle, nb_particles);
    std::cout << "End of writing" << std::endl;
    return 0;
}
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    /// Parsing options
    ///
    auto parser = cpp_tools::cl_parser::make_parser(loc_args::dimension{}, loc_args::dataType{}, loc_args::input_file{},
                                                    loc_args::output_file{}, cpp_tools::cl_parser::help{});

    // Parameter handling
    parser.parse(argc, argv);

    std::cout << cpp_tools::colors::blue << "Entering sort_particles...\n" << cpp_tools::colors::reset;

    const int dimension{parser.get<loc_args::dimension>()};
    std::cout << cpp_tools::colors::blue << "<params> dimension   : " << dimension << cpp_tools::colors::reset << '\n';
    const bool use_double = (parser.exists<loc_args::dataType>() ? false : true);
    std::cout << cpp_tools::colors::blue << "<params> data type   : " << (use_double ? "double" : "float")
              << cpp_tools::colors::reset << '\n';

    const auto input_file{parser.get<loc_args::input_file>()};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file  : " << input_file << cpp_tools::colors::reset
                  << '\n';
    }

    const auto output_file{parser.get<loc_args::output_file>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    bool verbose = true;
    if(dimension == 1)
    {
        if(use_double)
        {
            run<1, double>(input_file, output_file, verbose);
        }
        else
        {
            run<1, float>(input_file, output_file, verbose);
        }
    }
    else if(dimension == 2)
    {
        if(use_double)
        {
            run<2, double>(input_file, output_file, verbose);
        }
        else
        {
            run<2, float>(input_file, output_file, verbose);
        }
    }
    else if(dimension == 3)
    {
        if(use_double)
        {
            run<3, double>(input_file, output_file, verbose);
        }
        else
        {
            run<3, float>(input_file, output_file, verbose);
        }
    }
    else if(dimension == 4)
    {
        if(use_double)
        {
            run<4, double>(input_file, output_file, verbose);
        }
        else
        {
            run<4, float>(input_file, output_file, verbose);
        }
    }
    else
    {
        std::cerr << "Dimension should be lower than 5 (1,2,3,4)\n";
        throw std::invalid_argument("Dimension should be lower than 5 (1,2,3,4)\n");
    }

    //
    return 0;
}
