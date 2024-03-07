#include <iostream>
#include <limits>
#include <vector>

#include "parameters.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/grid_storage.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/utils/math.hpp"
#include "scalfmm/utils/sort.hpp"
#include "scalfmm/utils/tensor.hpp"
#include <cpp_tools/colors/colorized.hpp>

namespace localArg
{
    struct pbc
    {
        cpp_tools::cl_parser::str_vec flags = {"--per", "--pbc"};                       /*!< The flags */
        std::string description = "The periodicity in each direction (0 no periocity)"; /*!< The description
                                                                                         */
        std::string input_hint = "0,1,1";                                               /*!< The description */
        using type = std::vector<bool>;
    };
}   // namespace localArg
// template<std::size_t Dimension, typename CoordinatePointType , typename ArrayType>
// auto checkLimit(CoordinatePointType& coord, const ArrayType& period, const int &limite1d )-> bool {
//   using CoordinateType = typename CoordinatePointType::value_type;
//   bool check = true;
//   for (int d = 0; d < Dimension; ++d) {
//   //   std::cout << CoordinateType(0) << " <= " << coord[d] << " < "
//   //             << limite1d << std::endl;
//     if (period[d]) {
//       if (coord[d] < 0) {
//         coord[d] += limite1d;
//       } else if (coord[d] > limite1d - 1) {
//         coord[d] -= limite1d;
//       }
//     } else {
//       check =
//           check && scalfmm::math::between(coord[d], CoordinateType(0), CoordinateType(limite1d));
//     }
//   }
//   return check;
// }
// template<std::size_t Dimension>
// auto trueMorton(std::size_t index) ->std::size_t {
//     return (index > 3 ) ? index-((index>> Dimension)<< Dimension) : index ;

//   }

//////////////////////////////////////////////////////////////////////////
///
constexpr int dimension = 2;
using value_type = double;
using position_type = scalfmm::container::point<value_type, dimension>;
using box_type = scalfmm::component::box<position_type>;
//////////////////////////////////////////////////////////////////////////
auto main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) -> int
{
    //
    const bool CheckMortonIndex = false;   // check neighbors indexes in 2d
    const bool check_morton_m2l_list = true;
    const bool check_morton_m2l_list_per = false;
    // Parameter handling
    auto parser =
      cpp_tools::cl_parser::make_parser(cpp_tools::cl_parser::help{}, args::output_file(), /*args::tree_height{},*/
                                        args::order{}, localArg::pbc{},                    // args::thread_count{},
                                        args::block_size{}, args::log_file{}, args::log_level{});
    parser.parse(argc, argv);
    // Getting command line parameters
    const int tree_height{3 /*parser.get<args::tree_height>()*/};
    std::cout << cpp_tools::colors::blue << "<params> Tree height : " << tree_height << cpp_tools::colors::reset
              << '\n';
    //    const std::string input_file{parser.get<args::input_file>()};
    //    if(!input_file.empty())
    //    {
    //        std::cout << cpp_tools::colors::blue << "<params> Input file : " <<
    //        input_file << cpp_tools::colors::reset <<
    //        '\n';
    //    }
    std::vector<bool> pbc(dimension, false);
    if(parser.exists<localArg::pbc>())
    {
        pbc = parser.get<localArg::pbc>();
        if(pbc.size() != 2)
        {
            std::cerr << "Only works in 2 d \n";
            exit(-1);
        }
    }
    std::cout << "pbc: " << std::boolalpha;
    for(auto e: pbc)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;
    // build Box of size 1 center in O
    position_type box_center{0.5};
    box_type box(1.0, box_center);
    ///////////////////////////////////////////////////////////////////////////////////////
    ///
    /// dimension 2 check the morton indexes in periodic
    ///
    int level = 2;
    const int neighbour_separation = 1;

    // lambda function
    auto getN = [&level, &box](std::size_t indexRef)
    {

        auto pos = scalfmm::index::get_coordinate_from_morton_index<dimension>(indexRef);
        auto per = box.get_periodicity();
        auto neig1 = scalfmm::index::get_neighbors(pos, level, per, neighbour_separation);
        auto& indexes = std::get<0>(neig1);
        const auto& indexes_in_array1 = std::get<1>(neig1);
        const auto& nb = std::get<1>(neig1);
        //    std::cout << "Neighbors of  " << indexRef << " are ";
        //    for (std::size_t i = 0; i < nb; ++i) {
        //      std::cout << " " << indexes[i];
        //    }
        //    std::cout << '\n';
        std::sort(std::begin(indexes), std::begin(indexes) + nb);
        std::cout << cpp_tools::colors::blue << "sorted Neighbors of  " << indexRef << " are (" << nb << ")"
                  << cpp_tools::colors::reset << '\n';
        for(std::size_t i = 0; i < nb; ++i)
        {
            std::cout << " " << indexes[i];
        }
        std::cout << '\n';
        return indexes;
    };

    if(CheckMortonIndex)
    {
        // set the periodicity into the box
        for(int i = 0; i < dimension; ++i)
        {
            box.set_periodicity(i, pbc[i]);
        }
        std::cout << "box " << box << std::endl;
        // Check neighbors of cell indexRef at level
        std::size_t indexRef = 3;

        level = 2;
        getN(3);   // result 0 1 2 4 6 8 9 12
        ///
        /// periodicity
        ///      none 0,1,3
        ///      true,false  0, 1, 3, 10, 11
        ///      false,true, 0, 1, 3, 5, 7
        ///      true,true,  0, 1, 3, 5, 7, 10, 11, 15
        ///
        std::cout << cpp_tools::colors::blue << "\n No periodicity: result should be 0, 1, 3 \n"
                  << cpp_tools::colors::reset;
        for(int i = 0; i < dimension; ++i)
        {
            box.set_periodicity(i, false);
        }
        getN(0);
        box.set_periodicity(0, true);
        std::cout << cpp_tools::colors::blue << "\n periodicity in  X: result should be 0, 1, 3, 10, 11\n"
                  << cpp_tools::colors::reset;
        getN(0);
        std::cout << cpp_tools::colors::blue << "\n periodicity in  Y: result should be 0, 1, 3, 5, 7\n"
                  << cpp_tools::colors::reset;
        box.set_periodicity(0, false);
        box.set_periodicity(1, true);
        getN(0);
        std::cout << cpp_tools::colors::blue
                  << "\n full periodicity in  X and Y: result should be 0, 1, 3, 5, 7, 10, 11, 15\n"
                  << cpp_tools::colors::reset;
        box.set_periodicity(0, true);
        getN(0);
    }
    /////////////////////////////////////////////////////////////
    ///  Build a tree with periodicity
    ///
    // position_type box_center{0.5,0.5};
    // box_type box(1.0, box_center);
    box.set_periodicity(pbc);
    value_type box_width = 1.0;
    // construct equispaced points on a dimension grid
    const int N{int(std::pow(2, tree_height - 1))};
    value_type step{box_width / value_type(N)};

    std::cout << "N= " << N << " step " << step << std::endl;
    value_type start = box_center[0] - box_width / 2. + step * 0.5;
    value_type end = box_center[0] + box_width / 2. - step * 0.5;
    // The center in 1 dimension
    auto distrib1d =
      xt::linspace(start /*double(-box_width / 2.)*/, end /*double(box_width / 2.)*/, N /*number of points*/);
    auto particle_generator = scalfmm::tensor::generate_meshgrid<dimension>(distrib1d);
    std::cout << "Distrib 1d: ";
    for(int i = 0; i < distrib1d.size(); ++i)
    {
        std::cout << distrib1d[i] << " ";
    }
    std::cout << std::endl;
    //
    // Box 0
    // std::size_t  level = 2;
    // Lambda function to compute the shift on all neighbors of indexRef
    auto computeShift = [&level, &box, &distrib1d](std::size_t indexRef)
    {
        std::cout << " cell number " << indexRef << " at level " << level << std::endl;

        auto pbc = box.get_periodicity();
        auto box_width = box.width(0);
        // Compute centre of box 0
        auto pos = scalfmm::index::get_coordinate_from_morton_index<dimension>(indexRef);
        std::cout << " pos " << pos[0] << " " << pos[1] << std::endl;
        position_type center{distrib1d[pos[0]], distrib1d[pos[1]]};
        // Get sorted neighboors
        auto neig1 = scalfmm::index::get_neighbors(pos, level, pbc, neighbour_separation);
        auto& indexes = std::get<0>(neig1);
        const auto& nb = std::get<1>(neig1);
        std::sort(std::begin(indexes), std::begin(indexes) + nb);
        // Compute the shift for all  neighbors
        for(std::size_t i = 0; i < nb; ++i)
        {
            auto pos2 = scalfmm::index::get_coordinate_from_morton_index<dimension>(indexes[i]);
            std::cout << i << " morton " << indexes[i] << " pos " << pos2[0] << "  " << pos2[0] << std::endl;

            position_type centerNeig{distrib1d[pos2[0]], distrib1d[pos2[1]]};
            auto shift = scalfmm::index::get_shift(center, centerNeig, pbc, box_width);
            std::cout << "shift to apply on leaf " << indexes[i] << ": " << shift;
            std::cout << "  center: " << centerNeig << "  center+shift: " << centerNeig - shift << std::endl;
        }
    };
    std::cout << "Box: " << box << std::endl;
    //   getN(0);
    //   computeShift(0);
    //   std::cout <<"==================================================\n";
    //   //
    //     getN(5);
    // computeShift(5);
    //   std::cout <<"==================================================\n";
    //     //
    //   getN(15);
    //   computeShift(15);
    //   std::cout <<"==================================================\n";
    //////////////////
    // m2l(12) : 0 1 2 4 5 8 10 16 17 18 19 24 25 26 27 32 33 34 35 36 37 38 39 48 49 50 51
    // m2l(3)  : 5 7 10 11 13 14 15 20 21 22 23 28 29 30 31 40 41 42 43 44 45 46 47 60 61 62 63
    if(check_morton_m2l_list)
    {
        int level = 2;
        const int neighbour_separation = 1;
        std::cout << "\n\n ------------- M2L list -------------\n";
        std::size_t indexRef = 26;
        auto getNM2LI = [&level, &pbc](std::size_t indexRef)
        {
            std::cout << "-------------\n";
            auto pos = scalfmm::index::get_coordinate_from_morton_index<2>(indexRef);

            auto neig1 = scalfmm::index::get_m2l_list(pos, level, pbc, neighbour_separation);
            auto& indexes = std::get<0>(neig1);
            const auto& indexes_in_array1 = std::get<1>(neig1);
            const auto& nb = std::get<2>(neig1);
            std::cout << "Index " << indexRef << " List (index,pos) of size (" << nb << "): \n";
            for(std::size_t i = 0; i < nb; ++i)
            {
                std::cout << " (" << indexes[i] << ", " << indexes_in_array1[i] << ")";
            }
            std::cout << '\n';

            std::cout << '\n';
        };
        level = 4;
        getNM2LI(0);
        getNM2LI(3);
        getNM2LI(26);
        getNM2LI(15);
    }
    return 0;
}
