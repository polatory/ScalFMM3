

#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <list>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <cpp_tools/colors/colorized.hpp>
#include <scalfmm/container/variadic_adaptor.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

TEMPLATE_TEST_CASE("Variadic adaptor construction", "[variadic-adaptor-construction]", std::vector<float>)
{
    using namespace scalfmm;
    using variadic_type = container::variadic_adaptor<void, TestType, TestType, TestType>;
    using size_type = typename TestType::size_type;
    using value_type = typename TestType::value_type;

    SECTION("Default construction", "[default]")
    {
        variadic_type adaptor;
        auto sizes = adaptor.all_size();
        bool ok_sizes = std::apply([](auto const&... item) { return ((item == size_type{0}) && ...); }, sizes);
        REQUIRE(ok_sizes == true);
    }

    SECTION("Value construction", "[value-contructed]")
    {
        variadic_type single_value_construct(std::size_t{6}, value_type(9.));

        REQUIRE(single_value_construct.all_size() == typename variadic_type::size_type(6, 6, 6));
        std::for_each(std::begin(std::get<0>(single_value_construct)), std::end(std::get<0>(single_value_construct)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });
        std::for_each(std::begin(std::get<1>(single_value_construct)), std::end(std::get<1>(single_value_construct)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });
        std::for_each(std::begin(std::get<2>(single_value_construct)), std::end(std::get<2>(single_value_construct)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });
    }

    SECTION("Copy/move construction", "[copy-move-contructed]")
    {
        variadic_type to_copy_1(std::size_t{6}, value_type(9.));
        variadic_type to_copy_2(std::size_t{9}, value_type(51.3));

        variadic_type copied(to_copy_1);

        REQUIRE(copied.all_size() == typename variadic_type::size_type(6, 6, 6));
        std::for_each(std::begin(std::get<0>(copied)), std::end(std::get<0>(copied)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });
        std::for_each(std::begin(std::get<1>(copied)), std::end(std::get<1>(copied)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });
        std::for_each(std::begin(std::get<2>(copied)), std::end(std::get<2>(copied)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });

        variadic_type copied_equal = to_copy_2;

        REQUIRE(copied_equal.all_size() == typename variadic_type::size_type(9, 9, 9));
        std::for_each(std::begin(std::get<0>(copied_equal)), std::end(std::get<0>(copied_equal)),
                      [](auto const& e) { REQUIRE(e == value_type(51.3)); });
        std::for_each(std::begin(std::get<1>(copied_equal)), std::end(std::get<1>(copied_equal)),
                      [](auto const& e) { REQUIRE(e == value_type(51.3)); });
        std::for_each(std::begin(std::get<2>(copied_equal)), std::end(std::get<2>(copied_equal)),
                      [](auto const& e) { REQUIRE(e == value_type(51.3)); });

        variadic_type moved(std::move(to_copy_1));

        REQUIRE(moved.all_size() == typename variadic_type::size_type(6, 6, 6));
        std::for_each(std::begin(std::get<0>(moved)), std::end(std::get<0>(moved)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });
        std::for_each(std::begin(std::get<1>(moved)), std::end(std::get<1>(moved)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });
        std::for_each(std::begin(std::get<2>(moved)), std::end(std::get<2>(moved)),
                      [](auto const& e) { REQUIRE(e == value_type(9.)); });

        variadic_type moved_equal = std::move(to_copy_2);

        REQUIRE(moved_equal.all_size() == typename variadic_type::size_type(9, 9, 9));
        std::for_each(std::begin(std::get<0>(moved_equal)), std::end(std::get<0>(moved_equal)),
                      [](auto const& e) { REQUIRE(e == value_type(51.3)); });
        std::for_each(std::begin(std::get<1>(moved_equal)), std::end(std::get<1>(moved_equal)),
                      [](auto const& e) { REQUIRE(e == value_type(51.3)); });
        std::for_each(std::begin(std::get<2>(moved_equal)), std::end(std::get<2>(moved_equal)),
                      [](auto const& e) { REQUIRE(e == value_type(51.3)); });
    }

    SECTION("Iterators interface", "[iterators-interface]")
    {
        variadic_type data(std::size_t{6}, value_type(9.));
        // Adress
        REQUIRE(&(*std::get<0>(data).begin()) == &std::get<0>(*data.begin()));
        REQUIRE(&(*std::get<1>(data).begin()) == &std::get<1>(*data.begin()));
        REQUIRE(&(*std::get<2>(data).begin()) == &std::get<2>(*data.begin()));
        REQUIRE(&(*std::get<0>(data).cbegin()) == &std::get<0>(*data.cbegin()));
        REQUIRE(&(*std::get<1>(data).cbegin()) == &std::get<1>(*data.cbegin()));
        REQUIRE(&(*std::get<2>(data).cbegin()) == &std::get<2>(*data.cbegin()));
        // Value
        REQUIRE(*std::get<0>(data).begin() == std::get<0>(*data.begin()));
        REQUIRE(*std::get<1>(data).begin() == std::get<1>(*data.begin()));
        REQUIRE(*std::get<2>(data).begin() == std::get<2>(*data.begin()));
        REQUIRE(*std::get<0>(data).cbegin() == std::get<0>(*data.cbegin()));
        REQUIRE(*std::get<1>(data).cbegin() == std::get<1>(*data.cbegin()));
        REQUIRE(*std::get<2>(data).cbegin() == std::get<2>(*data.cbegin()));
        // Data access
        std::get<0>(*data.begin()) = value_type(1.);
        REQUIRE(std::get<0>(*data.begin()) == value_type(1.));

        // Adress
        REQUIRE(&(*(--std::get<0>(data).end())) == &std::get<0>(*(--data.end())));
        REQUIRE(&(*(--std::get<0>(data).end())) == &std::get<0>(*(--data.end())));
        REQUIRE(&(*(--std::get<0>(data).end())) == &std::get<0>(*(--data.end())));
        REQUIRE(&(*(--std::get<0>(data).cend())) == &std::get<0>(*(--data.cend())));
        REQUIRE(&(*(--std::get<0>(data).cend())) == &std::get<0>(*(--data.cend())));
        REQUIRE(&(*(--std::get<0>(data).cend())) == &std::get<0>(*(--data.cend())));
        // Value
        REQUIRE(*(--std::get<0>(data).end()) == std::get<0>(*(--data.end())));
        REQUIRE(*(--std::get<0>(data).end()) == std::get<0>(*(--data.end())));
        REQUIRE(*(--std::get<0>(data).end()) == std::get<0>(*(--data.end())));
        REQUIRE(*(--std::get<0>(data).cend()) == std::get<0>(*(--data.cend())));
        REQUIRE(*(--std::get<0>(data).cend()) == std::get<0>(*(--data.cend())));
        REQUIRE(*(--std::get<0>(data).cend()) == std::get<0>(*(--data.cend())));
        // Data access
        std::get<0>(*(--data.end())) = value_type(1.);
        REQUIRE(std::get<0>(*(--data.end())) == value_type(1.));
    }

    SECTION("Capacity interface", "[capacity-interface]")
    {
        variadic_type data(std::size_t{6}, value_type(9.));

        REQUIRE(std::get<0>(data).max_size() == std::get<0>(data.max_size()));
        REQUIRE(std::get<1>(data).max_size() == std::get<1>(data.max_size()));
        REQUIRE(std::get<2>(data).max_size() == std::get<2>(data.max_size()));

        data.resize(10);

        REQUIRE(size_type{10} == std::get<0>(data.all_size()));
        REQUIRE(size_type{10} == std::get<1>(data.all_size()));
        REQUIRE(size_type{10} == std::get<2>(data.all_size()));

        variadic_type to_resize{};
        to_resize.resize(10, value_type(8.));

        REQUIRE(size_type{10} == std::get<0>(to_resize.all_size()));
        REQUIRE(size_type{10} == std::get<1>(to_resize.all_size()));
        REQUIRE(size_type{10} == std::get<2>(to_resize.all_size()));
        std::for_each(std::begin(std::get<0>(to_resize)), std::end(std::get<0>(to_resize)),
                      [](auto const& e) { REQUIRE(e == value_type(8.)); });
        std::for_each(std::begin(std::get<1>(to_resize)), std::end(std::get<1>(to_resize)),
                      [](auto const& e) { REQUIRE(e == value_type(8.)); });
        std::for_each(std::begin(std::get<2>(to_resize)), std::end(std::get<2>(to_resize)),
                      [](auto const& e) { REQUIRE(e == value_type(8.)); });
    }

    SECTION("subscript interface", "[subscript-interface]")
    {
        variadic_type data(std::size_t{6}, value_type(9.));
        // Adress
        REQUIRE(&(*std::get<0>(data).begin()) == &std::get<0>(data[0]));
        REQUIRE(&(*std::get<1>(data).begin()) == &std::get<1>(data[0]));
        REQUIRE(&(*std::get<2>(data).begin()) == &std::get<2>(data[0]));
        REQUIRE(&(*std::get<0>(data).cbegin()) == &std::get<0>(data[0]));
        REQUIRE(&(*std::get<1>(data).cbegin()) == &std::get<1>(data[0]));
        REQUIRE(&(*std::get<2>(data).cbegin()) == &std::get<2>(data[0]));
        // Value
        REQUIRE(*std::get<0>(data).begin() == std::get<0>(data[0]));
        REQUIRE(*std::get<1>(data).begin() == std::get<1>(data[0]));
        REQUIRE(*std::get<2>(data).begin() == std::get<2>(data[0]));
        REQUIRE(*std::get<0>(data).cbegin() == std::get<0>(data[0]));
        REQUIRE(*std::get<1>(data).cbegin() == std::get<1>(data[0]));
        REQUIRE(*std::get<2>(data).cbegin() == std::get<2>(data[0]));
        // Data access
        std::get<0>(*data.begin()) = value_type(1.);
        REQUIRE(std::get<0>(data[0]) == value_type(1.));
    }
}

auto main(int argc, char* argv[]) -> int
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
