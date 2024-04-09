#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/utils.hpp"
#include <scalfmm/container/iterator.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/container/particle_container.hpp>
#include <scalfmm/meta/utils.hpp>

using namespace scalfmm;

auto main() -> int
{
    using particle_type = container::particle<double, 3, double, 2, double, 2, std::size_t>;
    container::particle_container<particle_type> c(13);
    auto it_traits = std::begin(c);
    using iterator_type = decltype(it_traits);
    std::cout << "--- traits: ---\n";
    std::cout << scalfmm::container::iterator_traits<iterator_type>::dimension << '\n';
    std::cout << scalfmm::container::iterator_traits<iterator_type>::inputs_size << '\n';
    std::cout << scalfmm::container::iterator_traits<iterator_type>::outputs_size << '\n';
    std::cout << "--- end traits ---\n";

    double inc{0};
    std::for_each(std::begin(c), std::end(c), [&inc](auto p) {
        p = particle_type(container::point<double>{inc, inc, inc}, inc * 2, inc * 3, std::size_t(inc)).as_tuple();
        ++inc;
    });

    std::for_each(std::begin(c), std::end(c), [](auto p) {
        std::cout << "-----\n";
        particle_type p_(p);
        std::cout << p_.position() << '\n';
        for(auto const& e: p_.inputs())
        {
            std::cout << e << '\n';
        }
        for(auto const& e: p_.outputs())
        {
            std::cout << e << '\n';
        }
        meta::repeat([](auto e) { std::cout << e << '\n'; }, p_.variables());
    });

    for(std::size_t i = 0; i < 13; ++i)
    {
        c.insert_position(i, container::point<double, 3>(0.));
        c.insert_inputs(i, std::make_tuple(18., 19.));
        c.insert_outputs(i, std::array{118., 119.});
        c.insert_variables(i, meta::to_tuple(std::array{2500}));
    }

    std::cout << "iterator\n";
    auto it = std::begin(c);
    for(std::size_t i = 0; i < 13; ++i)
    {
        std::cout << "-----\n";
        std::cout << particle_type(*it).position() << '\n';
        for(auto const& e: (particle_type(*it)).inputs())
        {
            std::cout << e << '\n';
        }
        for(auto const& e: (particle_type(*it)).outputs())
        {
            std::cout << e << '\n';
        }
        meta::repeat([](auto e) { std::cout << e << '\n'; }, particle_type(*it).variables());
        ++it;
    }

    std::for_each(std::begin(c), std::end(c), [](auto p) {
        std::cout << "-----\n";
        particle_type p_(p);
        std::cout << p_.position() << '\n';
        for(auto const& e: p_.inputs())
        {
            std::cout << e << '\n';
        }
        for(auto const& e: p_.outputs())
        {
            std::cout << e << '\n';
        }
        meta::repeat([](auto e) { std::cout << e << '\n'; }, p_.variables());
    });

    std::fill(container::position_begin(c), container::position_end(c),
              meta::to_tuple(typename particle_type::position_type(4.5)));
    std::for_each(std::begin(c), std::end(c), [](auto p) {
        std::cout << "-----\n";
        particle_type p_(p);
        std::cout << p_.position() << '\n';
    });

    auto p{particle_type(container::point<double>{0.1, 0.2, 0.3}, 0., 0., 1)};

    for(std::size_t i = 0; i < p.sizeof_dimension(); ++i)
        std::cout << p.position(i) << '\n';
    std::cout << "dimension : " << p.sizeof_dimension() << '\n';
    std::cout << "size of inputs : " << p.sizeof_inputs() << '\n';
    std::cout << "size of outputs : " << p.sizeof_outputs() << '\n';
    std::cout << "size of variables : " << p.sizeof_variables() << '\n';
    std::cout << container::particle_traits<decltype(p)>::dimension_size << '\n';
    std::cout << container::particle_traits<decltype(p)>::inputs_size << '\n';
    std::cout << container::particle_traits<decltype(p)>::outputs_size << '\n';
    std::cout << container::particle_traits<decltype(p)>::variables_size << '\n';

    std::cout << "position : " << p.position() << '\n';

    p.position() = 2 * p.position();

    std::cout << "position*2 : " << p.position() << '\n';

    p.position({1., 2., 3.});

    std::cout << "position update : " << container::position(p) << '\n';

    std::cout << "position[1] : " << p.position(1) << '\n';

    p.position(1) = 9.;

    std::cout << "position[1] new : " << p.position(1) << ' ' << container::position(p) << '\n';

    container::position(p)[1] = 9.3;
    std::cout << "position[1] new2 : " << p.position(1) << ' ' << container::position(p) << '\n';

    for(auto const& e: p.inputs())
    {
        std::cout << e << '\n';
    }

    for(auto& e: container::inputs(p))
    {
        e = 0.1;
    }

    for(auto const& e: p.inputs())
    {
        std::cout << e << '\n';
    }

    return 0;
}
