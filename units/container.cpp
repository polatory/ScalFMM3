//
// Units for containers
// --------------------
#define CATCH_CONFIG_RUNNER

#include <catch2/catch.hpp>
#include <iosfwd>
#include <iterator>
#include <tuple>
#include <vector>

#include <scalfmm/container/variadic_container.hpp>
#include <scalfmm/memory/aligned_allocator.hpp>
#include <scalfmm/tools/colorized.hpp>

#include <xsimd/xsimd.hpp>

TEMPLATE_TEST_CASE("Variadic container construction", "[variadic_container]", double, int)
{
    using namespace scalfmm;
    static const int alignment{XSIMD_DEFAULT_ALIGNMENT};
    constexpr std::size_t size_value{5};
    constexpr TestType default_value{TestType(2.9)};
    const std::vector<std::tuple<TestType>> cv(size_value, std::make_tuple(default_value));
    using var_t = container::variadic_container<TestType>;
    const var_t cv_var(size_value, default_value);

    // Contructors tests
    // =================

    SECTION("Default construction", "[default]")
    {
        var_t c;
        REQUIRE(c.size() == 0);
        REQUIRE(c.empty() == true);
    }

    SECTION("Sized construction", "[size-constructor-default]")
    {
        var_t c(size_value);
        REQUIRE(c.size() == size_value);
        REQUIRE(c.empty() == false);
    }

    SECTION("Sized construction with default value", "[size-constructor-default-value]")
    {
        var_t c(size_value, default_value);
        REQUIRE(c.size() == size_value);
        REQUIRE(c.empty() == false);
        for(auto t: c)
        {
            REQUIRE(t == std::make_tuple(default_value));
        }
    }

    SECTION("Iterator construction", "[iterator-constructor]")
    {
        var_t c(cv.begin(), cv.end());
        REQUIRE(c.size() == size_value);
        REQUIRE(c.empty() == false);
        for(auto&& t: c)
        {
            REQUIRE(t == std::make_tuple(default_value));
        }
    }

    SECTION("Copy construction", "[copy-constructor]")
    {
        var_t c(cv_var);
        REQUIRE(c.size() == size_value);
        REQUIRE(c.empty() == false);
        for(auto&& t: c)
        {
            REQUIRE(t == std::make_tuple(default_value));
        }
    }

    SECTION("Move construction", "[move-constructor]")
    {
        var_t c(var_t(size_value, default_value));
        REQUIRE(c.size() == size_value);
        REQUIRE(c.empty() == false);
        for(auto&& t: c)
        {
            REQUIRE(t == std::make_tuple(default_value));
        }
    }

    SECTION("Assignment operator", "[assignement-operator]")
    {
        var_t c(size_value), cc(size_value);
        c = cv_var;
        cc = var_t(size_value, default_value);
        REQUIRE(c.size() == size_value);
        REQUIRE(cc.size() == size_value);
        REQUIRE(c.empty() == false);
        REQUIRE(cc.empty() == false);
        for(auto&& t: c)
        {
            REQUIRE(t == std::make_tuple(default_value));
        }
        for(auto&& t: cc)
        {
            REQUIRE(t == std::make_tuple(default_value));
        }
    }

    SECTION("Allocator accessor", "[allocator-accessor]")
    {
        std::tuple<memory::aligned_allocator<alignment, TestType>> a;
        REQUIRE(cv_var.get_allocator() == a);
    }

    SECTION("At accessor", "[at-accessor]")
    {
        for(std::size_t i = 0; i < size_value; ++i)
        {
            REQUIRE(cv.at(i) == std::make_tuple(default_value));
        }
    }

    SECTION("Braket operator", "[bracket-operator]")
    {
        for(std::size_t i = 0; i < size_value; ++i)
        {
            REQUIRE(cv[i] == std::make_tuple(default_value));
        }
    }
}

// Quick fold expression
template<typename... Args>
bool all(Args... args)
{
    return (... && args);
}

TEST_CASE("Variadic vector access", "[vector-access]")
{
    using namespace scalfmm;
    constexpr std::tuple<double, float, int> ct(double{3.0}, float{2.0}, int{1});

    SECTION("Front and back accessor", "[front-back-access]")
    {
        container::variadic_container<std::tuple<double, float, int>> c(1, ct);

        REQUIRE(all(std::get<0>(c.front()) == std::get<0>(c.back()), std::get<0>(c.front()) == std::get<0>(ct),
                    std::get<0>(c.back()) == std::get<0>(ct)));
        REQUIRE(all(std::get<1>(c.front()) == std::get<1>(c.back()), std::get<1>(c.front()) == std::get<1>(ct),
                    std::get<1>(c.back()) == std::get<1>(ct)));
        REQUIRE(all(std::get<2>(c.front()) == std::get<2>(c.back()), std::get<2>(c.front()) == std::get<2>(ct),
                    std::get<2>(c.back()) == std::get<2>(ct)));
    }

    SECTION("Raw data accessor", "[raw-access]")
    {
        container::variadic_container<double, float, int> c(10, ct);

        REQUIRE(std::get<0>(c.data()) == &std::get<0>(c.front()));
        REQUIRE(std::get<1>(c.data()) == &std::get<1>(c.front()));
        REQUIRE(std::get<2>(c.data()) == &std::get<2>(c.front()));
        REQUIRE(*std::get<0>(c.data()) == std::get<0>(c.front()));
        REQUIRE(*std::get<1>(c.data()) == std::get<1>(c.front()));
        REQUIRE(*std::get<2>(c.data()) == std::get<2>(c.front()));
    }

    SECTION("Iterator access", "[iterator-access]")
    {
        container::variadic_container<double, float, int> c(10, ct);
        REQUIRE(std::distance(c.begin(), c.end()) == c.size());
        REQUIRE(std::distance(c.cbegin(), c.cend()) == c.size());
        REQUIRE(*c.begin() == ct);
        REQUIRE(*(c.end() - 1) == ct);
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
