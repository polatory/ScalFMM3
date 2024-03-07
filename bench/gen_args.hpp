#ifndef EXAMPLES_PARAMETERS_HPP
#define EXAMPLES_PARAMETERS_HPP

#include <cpp_tools/cl_parser/cl_parser.hpp>

namespace args
{
    struct tree_height : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--tree-height", "-th"};
        const char* description = "Tree height (or initial height in case of an adaptive tree).";
        using type = int;
        type def = 2;
    };

    struct order : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--order", "-o"};
        const char* description = "Precision setting.";
        using type = std::size_t;
        type def = 3;
    };

    struct thread_count
    {
        cpp_tools::cl_parser::str_vec flags = {"--threads", "-t"};
        const char* description = "Maximum thread count to be used.";
        using type = std::size_t;
        type def = 1;
    };

    struct input_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        const char* description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };

    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        const char* description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
    };

    struct log_level
    {
        cpp_tools::cl_parser::str_vec flags = {"--log-level", "-llog"};
        const char* description = "Log level to print.";
        using type = std::string;
        type def = "info";
    };

    struct log_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--log-file", "-flog"};
        const char* description = "Log to file using spdlog.";
        using type = std::string;
        type def = "";
    };

    struct block_size
    {
        cpp_tools::cl_parser::str_vec flags = {"--group-size", "-gs"};
        const char* description = "Group tree chunk size.";
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
    auto cli = cpp_tools::cl_parser::make_parser(tree_height{}, order{}, thread_count{}, input_file{}, output_file{},
                                                 block_size{}, log_file{}, log_level{}, cpp_tools::cl_parser::help{});
}   // namespace args

/**
 * \brief Store the PerfTest program parameters.
 */
struct command_line_parameters
{
    explicit command_line_parameters(const decltype(args::cli)& cli)
    {
        tree_height = cli.get<args::tree_height>();
        order = cli.get<args::order>();
        thread_count = cli.get<args::thread_count>();
        input_file = cli.get<args::input_file>();
        output_file = cli.get<args::output_file>();
        log_file = cli.get<args::log_file>();
        log_level = cli.get<args::log_level>();
        block_size = cli.get<args::block_size>();
    }

    int tree_height = 5;            ///< Tree height.
    std::size_t order = 3;          ///< Tree height.
    int thread_count = 1;           ///< Maximum thread count (when used).
    std::string input_file = "";    ///< Particles file.
    std::string output_file = "";   ///< Output particule file.
    std::string log_file = "";      ///< Log file.
    std::string log_level = "";     ///< Log file.
    int block_size = 250;           ///< Group tree group size
};

#endif   // EXAMPLES_PARAMETERS_HPP
