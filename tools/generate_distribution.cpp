/*
 * genarateDistributions.cpp
 *
 *  Created on: 14 September 2020
 *      Author: Olivier Coulaud
 */

//#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
//
#include "scalfmm/tools/data_generate.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/vtk_writer.hpp"
#include "scalfmm/utils/sort.hpp"
#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
/**
 * \file generate_distribution.cpp
 * \author { O. Coulaud }
 * \brief Generates points (non)uniformly distributed on a given geometry
 *
 * The goal of this driver is to generate uniform or non uniform points on the
 * following geometries
 *
 *   - Uniform : cube, cuboid, sphere, prolate,
 *   - Non uniform : ellipsoid
 *
 *  You can set two kind of physical values depending of your problem. By
 *   default all values are between 0 and 1.  If you select the argument -charge
 *   (see bellow) the values are between -1 and 1.  The arguments available are
 *
 * <b> General arguments:</b>
 *  <ul>
 * <li> --help (-h)    to see the parameters available in this driver</li>
 * <li>  --dim, --dimension    Dimension of the space (required)</li>
 * <li> --N, --n, --number_particles     The number of points in the distribution (required)</li>
 * <li> --in-val   The number of physical values associated to each particle (required)</li>
 * <li> --output-file, -fout value Output particle file (with extension .fma (ascii) or bfma (binary).
 * <li> --visu-file Filename for the visu file. vtp extension for the format (vtk)</li>
 *  </ul>
 * <b> Geometry arguments:</b>
 * <ul>
 * <li> --cuboid LX:LY:LZ  uniform distribution in a cube of size [0,LX]x[0,LY]x[0,LZ] </li>
 * <li> --sphere  R   uniform distribution on sphere of radius R </li>
 *
 * <li> --ball R uniform distribution in ball of radiusR <+li>
 * <li> --ellipsoid a:b:c  non uniform distribution on an ellipsoid center in 0 of aspect ratio
 *                   given by  a:b:c with a, b and c > 0  ex --ellipsoid 2:3:4</li>
 * <li> --unif-ellipsoid a:b:c uniform distribution on cuboid of size a:b:c center in 0.</li>
 * <li> --non-unif-ellipsoid  a:b:c< non uniform distribution on cuboid of size a:b:c center in 0.</li>
 * <li> --prolate a:c  ellipsoid with aspect ratio a:a:c center in 0 with c > a > 0</li>
 * <li> --plummer R (Highly non uniform) plummer distribution (astrophysics) (not yet) ex: --plummer 10 </li>
 * </ul>
 *
 * <b> Physical values argument:</b>
 *  <ul>
 * <li>  --randomInter std::string specify the interval of the uniform distribution or each input values. The default
 * is (0,1) <p>
 *      ex: --randomInter  a,b:c,d means uniform distribution on interval (a,b)  (resp. (c,d)) for variable 1 (resp.
 * 2) </li>
 * <li> --zero-mean, --zm  the average of the physical values is zero (flag) </li>
 * </ul>
 *
 * <b> other argument:</b>
 * <ul>
 *  <li> --extraLength value extra length to add to the boxWidth (default 0.0)</li>
 *  <li> --sort sort the particles according to a their morton index (level 20 in a tree) (flag)</li>
 *  <li> --use-float to generate a float distribution (not yet)</li>
 *  <li> --center  the center of the final distribution.</li>
 *  </ul>
 * <b> examples</b>
 *
 * \code
 *   Ex 1 - Generate a prolate distribution of 20000 points without input and save it in prolate.fma
 *
 *    generate_distribution --prolate 2:2:4   --dim 3 --n 20000 --in-val 0  -fout prolate.fma
 *
 *   Ex 2 Generate distribution of 100000 points in the cube [0,2]x [0,2]x[0,4] with one input with zero mean.
 *        Moreover the distribution is store in vtp format for visualization purposed (paraview or visit)
 *
 *   generate_distribution  --dim 3 --cuboid 2:2:4 --N 100000 -fout cuboid.bfma --visu-file cuboid.vtp --in-val 1  \
 *                        --zero-mean
 *
 *   Ex 3 here we generate 2 input values with random distribution between [10,30] et [-1,1]. Moreover the ellipsoid
 *          is centered in 3.0, 4.0, 2.0.
 *
 *    generate_distribution --dimension 3 --in-val 2 --unif-ellipsoid 1:3:5 --n 100000 --randomInter "10,30:-1,1"
 *               --output-file prolate.fma --visu-file prolate.vtp --centre 3,4,2
 *
 * \endcode
 */
namespace paramGenerate
{
    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
    };
    struct nbParticles : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--n", "--N", "--number_particles"};
        std::string description = "Number of particles to generate";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
    };
    struct nbPhysicalValues : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--in-val"};
        std::string description = "Number of physical values to generate";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
    };
    struct dimensionSpace : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dim", "--dimension", "-d"};
        std::string description = "Dimension of the space (1,2 or 3)";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
    };
    struct cuboid
    {
        cpp_tools::cl_parser::str_vec flags = {"--cuboid"};
        using type = std::string;
        std::string description = "Uniform distribution on cuboid of size a or a:b or a:b:c center in a/2:b/2:c/2";
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct ball
    {
        cpp_tools::cl_parser::str_vec flags = {"--ball"};
        using type = double;
        std::string description = "Uniform distribution in a ball (2d or 3d) of radius center R center in "
                                  "0";
        std::string input_hint = "double"; /*!< The input hint */
    };
    struct sphere
    {
        cpp_tools::cl_parser::str_vec flags = {"--sphere"};
        using type = double;
        std::string description = "Uniform distribution on sphere (2d, 3d) of radius center R center in 0";
        std::string input_hint = "double"; /*!< The input hint */
    };
    struct prolate
    {
        cpp_tools::cl_parser::str_vec flags = {"--prolate"};
        using type = std::string;
        std::string description = "Uniform distribution on a prolate cuboid of size  a:a:c center in 0"
                                  "--prolate a:c";
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct unifEllipsoid
    {
        cpp_tools::cl_parser::str_vec flags = {"--unif-ellipsoid"};
        using type = std::string;
        std::string description = "Uniform distribution on cuboid of size a:b:c center in 0";
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct nonUnifProlate
    {
        cpp_tools::cl_parser::str_vec flags = {"--non-unif-prolate"};
        using type = std::string;
        std::string description =
          "Non uniform distribution on ellipsoid of size a:a:b center in 0 (Points density ~ 10)\n"
          "";
        std::string input_hint = "a:b"; /*!< The input hint */
    };
    struct nonUnifEllipsoidDensity
    {
        cpp_tools::cl_parser::str_vec flags = {"--density"};
        using type = double;
        std::string description = "Density points for non uniform distribution on ellipsoid ";
        std::string input_hint = "value"; /*!< The input hint */
        type def = 10.0;
    };
    struct plummer
    {
        cpp_tools::cl_parser::str_vec flags = {"--plummer"};
        using type = double;
        std::string description = "Plummer distribution for points in a sphere of size R (--plummer R)";
        std::string input_hint = "double"; /*!< The input hint */
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
    struct zeromean
    {
        cpp_tools::cl_parser::str_vec flags = {"--zero-mean", "--zm"};
        std::string description = "The average of the all physical values is zero";
        using type = bool;
        enum
        {
            flagged
        };
    };
    struct charge
    {
        cpp_tools::cl_parser::str_vec flags = {"--charge", "--c"};
        std::string description = "if 1 generate physical values between -1 and 1"
                                  "otherwise generate between 0 and 1 (default 0)";
        using type = int;
        type def = 0;
        std::string input_hint = "int"; /*!< The input hint */
    };
    struct random
    {
        cpp_tools::cl_parser::str_vec flags = {"--randomInter"};
        using type = std::string;
        std::string description =
          "Uniform distribution on interval (a,b)  (resp. (c,d)) for variable 1 (resp. 2) ex: a:b-c:d ";
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct extraLength
    {
        cpp_tools::cl_parser::str_vec flags = {"--extraLength"};
        using type = double;
        std::string description = "Extra length to add to the box width (default 0;0) ";
        std::string input_hint = "double"; /*!< The input hint */
        type def = 0.0;
    };

    struct sort
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
    struct visu_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--visu-file"};
        std::string description = "Input filename (.vtp).";
        using type = std::string;
        std::string input_hint = "std::string"; /*!< The input hint */
    };
    struct centre
    {
        cpp_tools::cl_parser::str_vec flags = {"--centre", "-c"};         /*!< The flags */
        std::string description = "the center of the final distribution"; /*!< The description */
        std::string input_hint = "value,value,value";                     /*!< The description */
        using type = std::vector<double>;
    };
}   // namespace paramGenerate

template<typename VALUE_T, class VECTOR_T>
void sortData(const int& dimension, const VALUE_T width, std::vector<VALUE_T> a_centre, const int& nb_particles,
              VECTOR_T& particles1)
{
    if(dimension == 2)
    {
        using position_type = scalfmm::container::point<VALUE_T, 2>;
        using box_type = scalfmm::component::box<position_type>;
        position_type centre{a_centre[0], a_centre[1]};
        box_type box(width, centre);

        scalfmm::utils::sort_raw_array_with_morton_index(box, nb_particles, particles1);
        std::cout << "sort the data" << std::endl;
    }
    else if(dimension == 3)
    {
        using position_type = scalfmm::container::point<VALUE_T, 3>;
        using box_type = scalfmm::component::box<position_type>;
        position_type centre{a_centre[0], a_centre[1], a_centre[2]};
        box_type box(width, centre);

        scalfmm::utils::sort_raw_array_with_morton_index(box, nb_particles, particles1);
        std::cout << "sort the data" << std::endl;
    }
    else
    {
        std::cerr << "Sort option works only for dimension 2 and 3. (nothing done) " << std::endl;
    }
}

auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    /// Parsing options
    ///
    auto parser = cpp_tools::cl_parser::make_parser(
      cpp_tools::cl_parser::help{}, paramGenerate::nbParticles(), paramGenerate::nbPhysicalValues(),
      paramGenerate::dimensionSpace(), paramGenerate::dataType(), paramGenerate::cuboid(), paramGenerate::ball(),
      paramGenerate::sphere(), paramGenerate::prolate(), paramGenerate::unifEllipsoid(),
      paramGenerate::nonUnifProlate(), paramGenerate::nonUnifEllipsoidDensity(), paramGenerate::plummer(),
      paramGenerate::zeromean(), paramGenerate::random(), paramGenerate::sort(), paramGenerate::centre(),
      paramGenerate::extraLength(), paramGenerate::visu_file(), paramGenerate::output_file());

    // Parameter handling
    parser.parse(argc, argv);
    const int dimension{parser.get<paramGenerate::dimensionSpace>()};
    const int nb_particles{parser.get<paramGenerate::nbParticles>()};
    const int number_input_values{parser.get<paramGenerate::nbPhysicalValues>()};
    const int n_data_per_particle = dimension + number_input_values;
    const bool use_double = (parser.exists<paramGenerate::dataType>() ? false : true);
    //
    const auto output_file{parser.get<paramGenerate::output_file>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    //
    // GENERATE INPUT_DATA
    std::cout << " use_double " << std::boolalpha << use_double << std::endl;
    using value_type = double;
    std::vector<value_type> boxCenter(dimension, 0.0);   // centre of the data distribution
    value_type boxWith{};

    std::vector<value_type> data(nb_particles * n_data_per_particle, 0.0);
    //   generate_all_data(parser, data, dim,  number_input_values  );
    std::string aspectRatio;
    const int stride = dimension + number_input_values;
    if(parser.exists<paramGenerate::cuboid>())
    {
        aspectRatio = parser.get<paramGenerate::cuboid>();
        std::cout << "Genrate cuboid: " << aspectRatio << std::endl;
        // Aspect ratio
        int dim = 1 + std::count(aspectRatio.begin(), aspectRatio.end(), ':');
        // std::cout << "Dimension : " << dim << std::endl;
        std::vector<value_type> width(dim);
        std::replace(aspectRatio.begin(), aspectRatio.end(), ':', ' ');
        std::stringstream ss(aspectRatio);
        for(int i = 0; i < dimension; ++i)
        {
            value_type a;
            ss >> a;
            width[i] = a;
            boxWith = std::max(boxWith, width[i]);
            boxCenter[i] = 0.5 * width[i];
        }
        scalfmm::tools::uniform_points_in_cuboid(dimension, stride, data, width);
    }
    else if(parser.exists<paramGenerate::ball>())
    {
        if(!(dimension == 3 || dimension == 2))
        {
            std::cerr << "To generate a sphere the dimension should be 2 or 3" << std::endl;
            throw std::invalid_argument("To generate a sphere the dimension should be 3\n");
            ;
        }
        boxWith = parser.get<paramGenerate::ball>();
        scalfmm::tools::uniform_points_ball(dimension, stride, data, boxWith);
        boxWith *= 2.0;
    }
    else if(parser.exists<paramGenerate::sphere>())
    {
        boxWith = parser.get<paramGenerate::sphere>();   // the radius of the sphere
        scalfmm::tools::uniform_points_on_d_sphere(dimension, stride, data, boxWith);
        boxWith *= 2.0;
    }

    else if(parser.exists<paramGenerate::prolate>())
    {
        if(dimension != 3)
        {
            std::cerr << "To generate a prolate the dimension should be 3" << std::endl;
            throw std::invalid_argument("To generate a prolate the dimension should be 3\n");
        }
        // Aspect ratio
        aspectRatio = parser.get<paramGenerate::prolate>();
        std::array<value_type, 3> radius{};
        std::replace(aspectRatio.begin(), aspectRatio.end(), ':', ' ');
        std::cout << "aspectRatio " << aspectRatio << std::endl;
        std::stringstream ss(aspectRatio);
        for(int i = 0; i < 2; ++i)
        {
            value_type a;
            ss >> a;
            radius[i + 1] = a;
            boxWith = std::max(boxWith, radius[i]);
        }
        radius[0] = radius[1];

        boxWith *= 2.0;
        scalfmm::tools::uniform_points_on_ellipsoid(stride, radius, data);

        // scalfmm::tools::uniform_points_on_prolate(stride, radius, data);
    }
    else if(parser.exists<paramGenerate::unifEllipsoid>())
    {
        if(dimension != 3)
        {
            std::cerr << "To generate an ellipsoid the dimension should be 3" << std::endl;
            throw std::invalid_argument("To generate an ellipsoid the dimension should be 3\n");
        }
        // Aspect ratio
        aspectRatio = parser.get<paramGenerate::unifEllipsoid>();
        std::array<value_type, 3> radius;
        std::replace(aspectRatio.begin(), aspectRatio.end(), ':', ' ');
        std::stringstream ss(aspectRatio);
        std::cout << "radius: ";
        for(int i = 0; i < 3; ++i)
        {
            value_type a;
            ss >> a;
            radius[i] = a;
            boxWith = std::max(boxWith, radius[i]);
            std::cout << " " << radius[i];
        }
        std::cout << std::endl;
        boxWith *= 2.0;

        scalfmm::tools::uniform_points_on_ellipsoid(stride, radius, data);
    }
    else if(parser.exists<paramGenerate::nonUnifProlate>())
    {
        if(dimension != 3)
        {
            std::cerr << "To generate an ellipsoid the dimension should be 3" << std::endl;
            throw std::invalid_argument("To generate an ellipsoid the dimension should be 3\n");
        }
        // Aspect ratio
        aspectRatio = parser.get<paramGenerate::nonUnifProlate>();
        std::array<value_type, 2> radius;
        std::replace(aspectRatio.begin(), aspectRatio.end(), ':', ' ');
        std::cout << "aspectRatio " << aspectRatio << std::endl;
        std::stringstream ss(aspectRatio);
        for(int i = 0; i < 2; ++i)
        {
            value_type a;
            ss >> a;
            radius[i] = a;
            boxWith = std::max(boxWith, radius[i]);
        }
        boxWith *= 2.0;
        const auto density = parser.get<paramGenerate::nonUnifEllipsoidDensity>();
        scalfmm::tools::nonuniform_point_on_prolate(stride, radius, density, data);
    }
    else if(parser.exists<paramGenerate::plummer>())
    {
        if(dimension != 3)
        {
            std::cerr << "To generate PLummer distribution the dimension should be 3" << std::endl;
            throw std::invalid_argument("To generate PLummer distribution the dimension should be 3\n");
        }
        // Aspect ratio
        value_type boxWith = parser.get<paramGenerate::plummer>();
        scalfmm::tools::plummer_distrib(stride, boxWith, data);
        boxWith *= 2.0;
    }
    else
    {
        std::cerr << "No data distribution specified " << std::endl;
        throw std::invalid_argument("No data distribution specified !!\n");
    }
    //
    // GENERATE INPUT_DATA  (INPUT PHYSICAL VALES)
    bool zeromean = parser.exists<paramGenerate::zeromean>();
    std::vector<std::array<value_type, 2>> distrib(number_input_values);
    if(parser.exists<paramGenerate::random>())
    {
        aspectRatio = parser.get<paramGenerate::random>();
        std::cout << "aspectRatio: " << aspectRatio << std::endl;

        std::replace(aspectRatio.begin(), aspectRatio.end(), ',', ' ');
        std::cout << "aspectRatio: " << aspectRatio << std::endl;

        std::replace(aspectRatio.begin(), aspectRatio.end(), ':', ' ');
        std::cout << "aspectRatio " << aspectRatio << std::endl;
        std::stringstream ss(aspectRatio);
        for(int i = 0; i < number_input_values; ++i)
        {
            auto& ab = distrib[i];
            for(int j = 0; j < 2; ++j)
            {
                value_type a;
                ss >> a;
                ab[j] = a;
            }
        }
    }
    else
    {
        for(int i = 0; i < number_input_values; ++i)
        {
            auto& ab = distrib[i];
            ab[0] = 0.0;
            ab[1] = 1.0;
        }
    }
    for(int i = 0; i < number_input_values; ++i)
    {
        auto& ab = distrib[i];
        std::cout << i << " uniform distrib in [ " << ab[0] << ", " << ab[1] << "] " << std::endl;
    }

    if(parser.exists<paramGenerate::centre>())
    {
        // Move the distribution to the new center

        std::vector<double> centre = parser.get<paramGenerate::centre>();
        std::cout << "Centre: ";
        for(auto& e: centre)
        {
            std::cout << e << " ";
        }
        std::cout << std::endl;
        for(std::size_t i = 0; i < data.size(); i += stride)
        {
            for(int j = 0; j < dimension; ++j)
            {
                data[i + j] += value_type(centre[j]) - boxCenter[j];
            }
        }
        for(int j = 0; j < dimension; ++j)
        {
            boxCenter[j] = value_type(centre[j]);
        }
    }

    scalfmm::tools::generate_input_values(data, number_input_values, stride, distrib, zeromean);
    //
    boxWith += parser.get<paramGenerate::extraLength>();

    if(parser.exists<paramGenerate::sort>())
    {
        sortData(dimension, boxWith, boxCenter, nb_particles, data);
    }

    //  WRITE DATA
    std::cout << "boxWith: " << boxWith << std::endl;
    std::cout << "Write outputs in " << output_file << std::endl;
    if(use_double)
    {
        scalfmm::io::FFmaGenericWriter<value_type> writer(output_file);
        writer.writeHeader(boxCenter, boxWith, nb_particles, sizeof(value_type), n_data_per_particle, dimension,
                           number_input_values);
        writer.writeArrayOfReal(data.data(), n_data_per_particle, nb_particles);
    }
    else
    {
        scalfmm::io::FFmaGenericWriter<float> writer(output_file);
        writer.writeHeader(boxCenter, boxWith, nb_particles, sizeof(float), n_data_per_particle, dimension,
                           number_input_values);
        std::vector<float> dataFloat(data.size());
        std::copy(data.begin(), data.end(), dataFloat.begin());
        writer.writeArrayOfReal(dataFloat.data(), n_data_per_particle, nb_particles);
    }
    std::cout << "End of writing" << std::endl;

    if(parser.exists<paramGenerate::visu_file>())
    {
        if(dimension < 4)
        {
            std::string visufile(parser.get<paramGenerate::visu_file>());
            scalfmm::tools::io::exportVTKxml(visufile, data, dimension, number_input_values, nb_particles);
        }
        else
        {
            std::cerr << "vtp export works only for dimension 1, 2 or 3.\n";
        }
    }
}
