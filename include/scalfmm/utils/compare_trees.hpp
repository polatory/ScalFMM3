#pragma once
#include <iostream>

#include <xtensor-blas/xblas.hpp>

namespace scalfmm::utils
{
    /////////////////////////////////
    ///
    /**
     * @brief compare the cells of two trees
     *
     *  For all levels, depending on the option, we compare the multipole and local tensors. We calculate the
     * frobenius norm of the error between the two tensors. If this norm is smaller than eps, then the test is true.
     *
     *  option 1 only the multipoles
     *  option 2 only the locals
     *  option 3 both multipoles and locals
     *
     * @param tree1 first tree
     * @param tree2 second tree
     * @param eps the threshold
     * @param option int (1,2,3) -the option describe above
     * @return the comparaison
     */
    template<typename Tree1, typename Tree2, typename Value_type>
    auto inline compare_two_trees(Tree1 const& tree1, Tree2 const& tree2, Value_type const eps, int option) -> bool
    {
        bool check{true}, check_mul{true}, check_loc{true};

        check = (tree1.height() == tree2.height()) and (tree1.order() == tree2.order());
        if(not check)
        {
            std::cout << "Wrong height or order\n";
            return check;
        }
        check = (tree1.group_of_cell_size() == tree2.group_of_cell_size());
        if(not check)
        {
            std::cout << "Wrong cell group sizer\n";
            return check;
        }
        check = (tree1.group_of_leaf_size() == tree2.group_of_leaf_size());
        if(not check)
        {
            std::cout << "Wrong leaf group sizer\n";
            return check;
        }

        for(std::size_t level = tree1.leaf_level(); level >= tree1.top_level(); --level)
        {
            auto start1 = tree1.begin_mine_cells(level);
            auto start2 = tree2.begin_mine_cells(level);
            for(auto grp1 = start1, grp2 = start2; grp1 != tree1.end_mine_cells(level); ++grp1, ++grp2)
            {
                if((*grp1)->size() != (*grp2)->size())
                {
                    std::cout << "wrong group size for group index1" << std::distance(start1, grp1) << " in tree 1";
                    std::cout << " and index2 " << std::distance(start2, grp2) << " in tree 2 \n";
                    return false;
                }
                for(std::size_t index = 0; index < (*grp1)->size(); ++index)
                {
                    auto const& cell1 = (*grp1)->ccomponent(index);
                    auto const& cell2 = (*grp2)->ccomponent(index);
                    if(cell1.csymbolics().morton_index != cell2.csymbolics().morton_index)
                    {
                        std::cerr << "wrong morton index - index1 " << cell1.csymbolics().morton_index << "  index2 "
                                  << cell2.csymbolics().morton_index << std::endl;
                        return false;
                    }
                    // Multipoles
                    if(option != 2)
                    {
                        auto mults1 = cell1.cmultipoles();
                        auto mults2 = cell2.cmultipoles();
                        auto number_of_arrays = mults1.size();
                        for(std::size_t l = 0; l < number_of_arrays; ++l)
                        {
                            auto& mult1 = mults1.at(l);
                            auto& mult2 = mults2.at(l);
                            auto diff = xt::abs(mult1 - mult2);
                            auto error = xt::linalg::norm(diff, xt::linalg::normorder::frob);
                            check = (error < eps);
                            if(not check)
                            {
                                std::cerr << "level " << level << " Cell morton " << cell1.csymbolics().morton_index
                                          << " error" << error << std::endl;
                                std::cerr << "mult1(" << l << ")\n" << mult1 << std::endl;
                                std::cerr << "mult2(" << l << ")\n" << mult2 << std::endl;
                                check_mul = false;
                            }
                        }
                    }
                    // locals
                    if(option != 1)
                    {
                        auto const& locals1 = cell1.clocals();
                        auto const& locals2 = cell2.clocals();
                        auto number_of_arrays = locals1.size();

                        for(std::size_t l = 0; l < number_of_arrays; ++l)
                        {
                            auto const& local1 = locals1.at(l);
                            auto const& local2 = locals2.at(l);
                            auto diff = xt::abs(local1 - local2);
                            auto error = xt::linalg::norm(diff, xt::linalg::normorder::frob);
                            check = (error < eps);
                            if(not check)
                            {
                                std::cerr << "level " << level << " Cell morton " << cell1.csymbolics().morton_index
                                          << " error " << error << std::endl;
                                std::cerr << "local1(" << l << ")\n" << local1 << std::endl;
                                std::cerr << "local2(" << l << ")\n" << local2 << std::endl;
                                check_loc = false;
                            }
                        }
                    }
                }
            }
        }
        return check and check_mul and check_loc;
    }
}   // namespace scalfmm::utils
