// @FUSE_OMP


#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include <scalfmm/container/iterator.hpp>
//
//
#include "scalfmm/meta/type_pack.hpp"
//#include "scalfmm/meta/utils.hpp"
//
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/matrix_kernels/scalar_kernels.hpp"
//
#include "scalfmm/algorithms/full_direct.hpp"
//
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tools/laplace_tools.hpp"
#include "scalfmm/utils/parameters.hpp"
//

#include <chrono>
#include <iostream>
#include <string>

#include <array>
#include <tuple>

#include <cpp_tools/cl_parser/help_descriptor.hpp>
#include <cpp_tools/cl_parser/tcli.hpp>
#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/timers/simple_timer.hpp>

/// \file  DirectComputation.cpp
//!
//! \brief DirectComputation: Driver to compute direct interaction between N
//! particles for 1/r kernel.
//!
//! DirectComputation: Driver to compute direct interaction between N particles
//! for 1/r kernel. the particles are read from file given by -fin argument and
//! potential, forces are stored in FMA format.
//!  <b> General arguments:</b>
//!     \param   -help (-h)      to see the parameters available in this driver
//!     \param   -fin name:  file name  to convert (with extension .fma (ascii)
//!     or bfma (binary).
//!                             Only our FMA (.bma, .bfma) is allowed "
//!     \param    -fout filenameOUT   output file  with extension (default
//!     output.bfma) \param   -verbose : print index x y z Q V fx fy fz
//! \code
//! USAGE:
//!  ./examples/RelWithDebInfo/direct_computation [--help] --input-file value [--output-file value] [--log-file value]
//!  [--log-level value] [--kernel value] [--dimension value] [--post_traitement]
//!
//! DESCRIPTION:
//!
//!   --help, -h
//!       Display this help message
//!
//!    --input-file, -fin value
//!        Input filename (.fma or .bfma).
//!
//!    --output-file, -fout value
//!        Output particle file (with extension .fma (ascii) or bfma (binary).
//!
//!    --log-file, -flog value
//!        Log to file using spdlog.
//!
//!    --log-level, -llog value
//!        Log level to print.
//!
//!    --kernel, --k value
//!        Matrix kernels:
//!        0 1/r, 1) grad(1/r), 2) p & grad(1/r) 3) mrhs,
//!        4) 1/r^2
//!        5) ln (2d)
//!
//!    --dimension, --d value
//!       Dimension : \n  -  1 <dimension <4
//!
//!   --post_traitement, --pt`
//!         Post traitement to obtain Electric field or the weight
//!/endcode
//! <b>examples</b>
//!  * Onve over r2 in dimension 2
//!  \code
//!   ./direct_computation --input-file  input_2d_file.fma  --kernel 4 --dimension 2  --output-file res.fma
//!  \endcode
//! file-name.{fma, bfma} \endcode
//
namespace local_args
{
    // struct input_file : cpp_tools::cl_parser::required_tag
    // {
    //     cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
    //     std::string description = "Input filename (.fma or .bfma).";
    //     using type = std::string;
    // };
    // struct output_file
    // {
    //     cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
    //     std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
    //     using type = std::string;
    // };
    struct newmatrix_kernel : public laplace::args::matrix_kernel
    {
        std::string description = "Matrix kernels: \n   0 1/r, 1) grad(1/r), 2) p & grad(1/r) 3) mrhs,\n"
                                  "4) 1/r^2  5) ln in 2d ";
    };
    struct dimension : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "--d"};
        std::string description = "Dimension : \n  -  1 <=dimension <=4";
        using type = int;
        type def = 1;
    };
}   // namespace local_args

template<int dimension, typename value_type, class matrix_kernel>
auto direct_run(const std::string& input_file, const std::string& output_file, const bool postreat) -> int
{
    std::cout << cpp_tools::colors::blue << "Entering tree test...\n" << cpp_tools::colors::reset;

    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                  << '\n';
    }

    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    //  The matrix kernel
    static constexpr std::size_t nb_inputs{matrix_kernel::km};
    static constexpr std::size_t nb_outputs{matrix_kernel::kn};
    //
    matrix_kernel mk{};
    //
    // Open particle file

    cpp_tools::timers::timer<std::chrono::minutes> time{};

    using point_type = scalfmm::container::point<value_type, dimension>;
    using particle_type =
      scalfmm::container::particle<value_type, dimension, value_type, nb_inputs, value_type, nb_outputs>;

    // Construct the container of particles
    using container_type = scalfmm::container::particle_container<particle_type>;
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << cpp_tools::colors::green << "Creating & Inserting ...\n" << cpp_tools::colors::reset;
    //
    value_type box_width{};
    point_type box_center{};
    time.tic();
    container_type* container{};
    // read the data (pos and inputs). The outputs are set to zeros.
    laplace::read_data<dimension>(input_file, container, box_center, box_width);
    time.tac();
    const std::size_t number_of_particles{container->size()};

    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::green << "Box center = " << box_center << " box width = " << box_width
              << cpp_tools::colors::reset << '\n';
    std::cout << cpp_tools::colors::yellow << "Container loaded in " << time.elapsed() << " s\n"
              << cpp_tools::colors::reset;

    std::cout << cpp_tools::colors::green << "full interaction computation ...\n" << cpp_tools::colors::reset;
    time.tic();

    //scalfmm::algorithms::full_direct(std::begin(*container), std::end(*container), mk);
    scalfmm::algorithms::full_direct(*container, mk);

    time.tac();
    std::cout << cpp_tools::colors::green << "... Done.\n" << cpp_tools::colors::reset;
    std::cout << cpp_tools::colors::yellow << "Computation done in " << time.elapsed() << " min\n"
              << cpp_tools::colors::reset;

    if(postreat)
    {
        laplace::post_traitement(mk, *container);
    }

    if(!output_file.empty())
    {
        std::cout << "Write outputs in " << output_file << std::endl;
        scalfmm::io::FFmaGenericWriter<value_type> writer(output_file);
        writer.writeDataFrom(*container, number_of_particles, box_center, box_width);
    }
    return 0;
}
template <typename value_type>
auto test_one_over_r2(const int dimension, const std::string& input_file, const std::string& output_file,
                      const bool postreat) -> int
{
    using matrix_kernel = scalfmm::matrix_kernels::others::one_over_r2;
    if(dimension == 1)
    {
        direct_run<1, value_type, matrix_kernel>(input_file, output_file, postreat);
    }
    else if(dimension == 2)
    {
        direct_run<2, value_type, matrix_kernel>(input_file, output_file, postreat);
    }
    else if(dimension == 3)
    {
        direct_run<3, value_type, matrix_kernel>(input_file, output_file, postreat);
    }
    else if(dimension == 4)
    {
        direct_run<4, value_type, scalfmm::matrix_kernels::others::one_over_r2>(input_file, output_file, postreat);
    }
    else
    {
        std::cerr << "Dimension should be lower than 5 (d < 5)\n";
        std::exit(EXIT_FAILURE);
    }
    return 0;
}
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    using value_type = double;

    //
    // Parameter handling
    auto parser = cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::input_file(),
                                                    args::output_file(), local_args::newmatrix_kernel{},
                                                    local_args::dimension{}, laplace::args::post_traitement{});
    parser.parse(argc, argv);
    const std::string input_file{parser.get<args::input_file>()};
    if(!input_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Input file : " << input_file << cpp_tools::colors::reset
                  << '\n';
    }

    const auto output_file{parser.get<args::output_file>()};
    if(!output_file.empty())
    {
        std::cout << cpp_tools::colors::blue << "<params> Output file : " << output_file << cpp_tools::colors::reset
                  << '\n';
    }
    const bool postreat(parser.exists<laplace::args::post_traitement>());
    const int matrix_type = parser.get<local_args::newmatrix_kernel>();
    const int dimension = parser.get<local_args::dimension>();
    // if(dimension > )
    //
    switch(matrix_type)
    {
    case 0:
        using matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
        if(dimension == 1)
        {
            direct_run<1, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        else if(dimension == 2)
        {
            direct_run<2, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        else
        {
            direct_run<3, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        break;
    case 1:
        if(dimension == 1)
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<1>;
            direct_run<1, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        else if(dimension == 2)
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<2>;
            direct_run<2, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        else
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::grad_one_over_r<3>;
            direct_run<3, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        break;
    case 2:
        if(dimension == 1)
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<1>;
            direct_run<1, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        else if(dimension == 2)
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<2>;
            direct_run<2, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        else
        {
            using matrix_kernel_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<3>;
            direct_run<3, value_type, matrix_kernel_type>(input_file, output_file, postreat);
        }
        break;
    case 3:
        direct_run<3, value_type, scalfmm::matrix_kernels::laplace::like_mrhs>(input_file, output_file, postreat);
        break;
    case 4:
        test_one_over_r2<value_type>(dimension, input_file, output_file, postreat);
        break;
    case 5:
        direct_run<2, value_type, scalfmm::matrix_kernels::laplace::ln_2d>(input_file, output_file, postreat);
        break;
    default:
        std::cout << "Kernel not implemented. values are\n Laplace kernels: 0) 1/r, 1) grad(1/r),"
                  << " 2) p + grad(1/r) 3) like_mrhs." << std::endl
                  << "Scalar kernels 4) 1/r^2 5) ln in 2d" << std::endl;
    }
}
