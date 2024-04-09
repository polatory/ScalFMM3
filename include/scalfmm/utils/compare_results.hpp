// See LICENCE file at project root
#ifndef SCALFMM_UTILS_COMPARE_RESULTS_HPP
#define SCALFMM_UTILS_COMPARE_RESULTS_HPP

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/accurater.hpp"
#include "scalfmm/utils/sort.hpp"

namespace scalfmm
{
    namespace utils
    {
        /////////////////////////////////
        ///
        ///
        template<class array_type>
        void compare_two_arrays(const std::string& tag, const int dimension, const std::size_t& nbParticles,
                                const std::vector<int>& index1_to_compare, const std::vector<int>& index2_to_compare,
                                const array_type& array1, const array_type& array2)
        {
            //
            using value_type = typename array_type::value_type;
            int nb_val_per_part1 = static_cast<int>(array1.size() / nbParticles);
            int nb_val_per_part2 = static_cast<int>(array2.size() / nbParticles);
            int nb_index = static_cast<int>(index1_to_compare.size());
            accurater<value_type>* error = nullptr;
            error = new accurater<value_type>[nb_index];
            auto max_idx = *std::max_element(index1_to_compare.begin(), index1_to_compare.end());
            if(max_idx >= nb_val_per_part1)
            {
                std::cout << "Wrong value for index1 " << max_idx << " Should be < " << nb_val_per_part1 << std::endl;
                std::exit(EXIT_FAILURE);
            }
            max_idx = *std::max_element(index2_to_compare.begin(), index2_to_compare.end());
            if(max_idx >= nb_val_per_part2)
            {
                std::cout << "Wrong value for index2 " << max_idx << " Should be < " << nb_val_per_part2 << std::endl;
                std::exit(EXIT_FAILURE);
            }
            for(std::size_t idxPart = 0; idxPart < nbParticles; ++idxPart)
            {
                auto pos1 = idxPart * nb_val_per_part1;
                auto pos2 = idxPart * nb_val_per_part2;
                bool samePos = true;
                value_type errorPos{};
                for(int i = 0; i < dimension; ++i)
                {
                    samePos = samePos && (array1[pos1 + i] == array2[pos2 + i]);
                    errorPos += (array1[pos1 + i] - array2[pos2 + i]) * (array1[pos1 + i] - array2[pos2 + i]);
                }
                if(errorPos > 1.e-13)
                {
                    std::cerr << "Wrong positions  error: " << std::sqrt(errorPos) << std::endl << "   P1: ";
                    for(int i = 0; i < dimension; ++i)
                    {
                        std::cerr << array1[pos1 + i] << " ";
                    }
                    std::cerr << std::endl << "   P2: ";
                    for(int i = 0; i < dimension; ++i)
                    {
                        std::cerr << array2[pos2 + i] << " ";
                    }
                    std::cerr << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                for(int idxVal = 0; idxVal < nb_index; ++idxVal)
                {
                    error[idxVal].add(array1[pos1 + index1_to_compare[idxVal]],
                                      array2[pos2 + index2_to_compare[idxVal]]);
                    //                    std::cout << idxPart << " " << idxVal << "  " << array1[pos1 +
                    //                    index1_to_compare[idxVal]] << "   "
                    //                              << array2[pos2 + index2_to_compare[idxVal]] << "  diff "
                    //                              << array1[pos1 + index1_to_compare[idxVal]] - array2[pos2 +
                    //                              index2_to_compare[idxVal]]
                    //                              << std::endl;
                }
            }
            // Print for information
            for(int idxVal = 0; idxVal < nb_index; ++idxVal)
            {
                std::cout << tag << " index  " << idxVal << " " << error[idxVal] << std::endl;
            }
        }

    }   // namespace utils
}   // namespace scalfmm
#endif   // SCALFMM_UTILS_COMPARE_RESULTS_HPP
