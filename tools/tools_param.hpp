#ifndef TOOLS_PARAMETERS_HPP
#define TOOLS_PARAMETERS_HPP

#include <cpp_tools/cl_parser/cl_parser.hpp>

namespace args_tools
{
    struct dimensionSpace : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--dim", "--dimension", "-d"};
        std::string description = "Dimension of the space (1,2 or 3)";
        using type = int;
        std::string input_hint = "int"; /*!< The input hint */
    };
    struct tree_height : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--tree-height", "-th"};
        std::string description = "Tree height (or initial height in case of an adaptive tree).";
        using type = int;
        type def = 2;
    };

    struct order : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--order", "-o"};
        std::string description = "Precision setting.";
        using type = std::size_t;
        type def = 3;
    };

    struct thread_count
    {
        cpp_tools::cl_parser::str_vec flags = {"--threads", "-t"};
        std::string description = "Maximum thread count to be used.";
        using type = std::size_t;
        type def = 1;
    };

    struct input_file_req : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input particles filename (.fma or .bfma).";
        using type = std::string;
    };
    struct input_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };

    struct block_size
    {
        cpp_tools::cl_parser::str_vec flags = {"--group-size", "-gs"};
        std::string description = "Group tree chunk size.";
        using type = int;
        type def = 250;
    };

    struct pbc
    {
        cpp_tools::cl_parser::str_vec flags = {"--per", "--pbc"};                         /*!< The flags */
        std::string description = "The periodicity in each direction (0 no periodicity)"; /*!< The description
                                                                                           */
        std::string input_hint = "0,1,1";                                                 /*!< The description */
        using type = std::vector<bool>;
    };

    struct extended_tree_height
    {
        using type = int;
        cpp_tools::cl_parser::str_vec flags = {"--ext-tree-height", "-eth"}; /*!< The flags */
        std::string description =
          "The number of level above the root level (used for periodicity)"; /*!< The description
                                                                              */
        std::string input_hint = "3";
        type def = 0;   // defaut no level for non periodic simulation
    };
 
}
#endif   // TOOLS_PARAMETERS_HPP
