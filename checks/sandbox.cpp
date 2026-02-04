//---------------------
// Experimental example
//---------------------
#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <inria/tcli/help_descriptor.hpp>
#include <inria/tcli/tcli.hpp>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/algorithm/generate.hpp>
#include <range/v3/algorithm/result_types.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/core.hpp>
#include <range/v3/functional/concepts.hpp>
#include <range/v3/iterator/concepts.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/range_fwd.hpp>
#include <range/v3/view/generate.hpp>

#include <xsimd/xsimd.hpp>
#include <xtensor/xtensor.hpp>

#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <scalfmm/container/point.hpp>
#include <scalfmm/functional/utils.hpp>
#include <scalfmm/interpolation/uniform.hpp>
#include <scalfmm/tools/colorized.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/tree.hpp>

namespace args
{
    struct nb_particle : inria::tcli::required_tag
    {
        inria::tcli::str_vec flags = {"--n-particles", "-nbp"};
        std::string description = "Number of particles to generate.";
        using type = std::size_t;
    };

    struct tree_height : inria::tcli::required_tag
    {
        inria::tcli::str_vec flags = {"--tree-height", "-th"};
        std::string description = "Tree height (or initial height in case of an adaptive tree).";
        using type = std::size_t;
    };

    struct thread_count : inria::tcli::required_tag
    {
        inria::tcli::str_vec flags = {"--threads", "-t"};
        std::string description = "Maximum thread count to be used.";
        using type = std::size_t;
        type def{1};
    };

    struct order : inria::tcli::required_tag
    {
        inria::tcli::str_vec flags = {"--order", "-o"};
        std::string description = "Precision order.";
        using type = std::size_t;
    };

    struct file : inria::tcli::required_tag
    {
        inria::tcli::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };

    struct chunk_size
    {
        inria::tcli::str_vec flags = {"--group-size", "-gs"};
        std::string description = "Group tree chunk size.";
        using type = std::size_t;
        type def{250};
    };

    auto cli = inria::tcli::make_parser(nb_particle{}, tree_height{}, thread_count{}, file{}, inria::tcli::help{},
                                        chunk_size{}, order{});
}   // namespace args

/**
 * \brief Store the PerfTest program parameters.
 */
struct params
{
    std::size_t m_nb_particle{};   ///< Tree height.
    std::size_t m_tree_height{};   ///< Tree height.
    std::size_t m_nb_threads{};    ///< Maximum thread count (when used).
    std::string m_filename{};      ///< Particles file.
    std::size_t m_group_size{};    ///< Group tree group size
    std::size_t m_order{};         ///< Group tree group size

    params() = default;
    params(const decltype(args::cli)& cli)
    {
        m_order = cli.get<args::order>();
        m_nb_particle = cli.get<args::nb_particle>();
        m_tree_height = cli.get<args::tree_height>();
        m_nb_threads = cli.get<args::thread_count>();
        m_filename = cli.get<args::file>();
        m_group_size = cli.get<args::chunk_size>();
    }
};

using namespace scalfmm;

CPP_template(class Iter, class Sent,
             class Fun)(requires   // ranges::Invocable<Fun&>  &&
                                   // ranges::OutputIterator<Iter,ranges::invoke_result_t<Fun&>> &&
                                   // ranges::Sentinel<Sent, Iter>
                        ranges::Writable<Iter, ranges::invoke_result_t<Fun&>>) void my_algorithm(Iter first, Sent last,
                                                                                                 Fun f)
{
    // ...
}

// Quick type displayer
template<typename T>
class TD;

int main(int argc, char** argv)
{
    args::cli.parse(argc, argv);
    params parameters(args::cli);

    interpolation::uniform_interpolator<double, 3> unif_interpolator(parameters.m_order, parameters.m_tree_height);

    // variad<int,float> ot{{8,9},{1.3,2.6}};

    // variad<int, float> copy{ot};
    // std::cout << colors::blue << std::get<0>(copy)[0] << " " << colors::green
    // << std::get<1>(copy)[0] << colors::reset << '\n';

    // variad<int, float, double> v_{{1,2}, {3.3,4.4}, {5.5,6.6}};
    // variad<int, float, double> v_(10);
    // v_.push_back(std::make_tuple(2,1.1,2.2));
    // std::cout << std::get<0>(v_).size() << '\n';
    // std::cout << colors::blue << std::get<0>(v_)[0] << " " << colors::green <<
    // std::get<1>(v_)[0] << " " << colors::yellow << std::get<2>(v_)[0] <<
    // colors::reset << '\n';

    // const std::size_t tree_height{parameters.m_tree_height};

    // using source_t = container::particle<double,3,double,std::size_t>;
    // using position_t = container::point<double,3>;
    // using container_source_t = container::particle_container<source_t>;

    // const std::size_t nb_of_part{parameters.m_nb_particle};

    // container_source_t cs{nb_of_part};

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<double> dis(-1.0, 1.0);
    // auto random_r = [&dis, &gen](){ return dis(gen); };

    // auto make_particle = [&tree_height, &random_r]()
    //{
    //  position_t pos = {random_r(), random_r(), random_r()};
    //  source_t part( pos
    //               , random_r()
    //               , index::get_morton_index(pos,
    //               component::box<position_t>{2.0, {0.0,0.0,0.0}}, tree_height)
    //               );
    //  return part.as_tuple();
    //};

    // auto print_particle = [](source_t p)
    //{
    //  std::cout << colors::yellow << "=================" << colors::reset <<
    //  '\n'; std::cout << "x = " << std::get<0>(p) << '\n'; std::cout << "y = "
    //  << std::get<1>(p) << '\n'; std::cout << "z = " << std::get<2>(p) << '\n';
    //  std::cout << "p = " << std::get<3>(p) << '\n';
    //  std::cout << "x.p = " << p.position()[0] << '\n';
    //  std::cout << "norm2 = " << utils::norm2(p.position()) << '\n';
    //  std::cout << "morton = " << std::get<4>(p) << '\n';

    //};

    // auto pred = [](auto a, auto b)
    //{
    //  return std::get<4>(a) < std::get<4>(b);
    //};

    // std::vector<std::tuple<int,float>> v = {{8,3.0},{9,2.1},{10,9.6}};

    // my_algorithm(cs.begin(), cs.end(), make_particle);

    // std::generate(cs.begin(), cs.end(), make_particle);
    // std::sort(cs.begin(), cs.end(), pred);
    // std::for_each(cs.begin(), cs.end(), print_particle);

    // component::tree t{};
    return 0;
}
