//
// Units for containers
// --------------------
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <scalfmm/tools/colorized.hpp>
#include <string>

#define SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT
#include <scalfmm/utils/static_assert_as_exception.hpp>
// Followed by file where assertions must be tested
#include <scalfmm/container/point.hpp>

TEST_CASE("Static test", "[static-assertion]")
{
    using namespace scalfmm;

    SECTION("Assertion test", "[assertion]")
    {
        bool throwed = false;
        try
        {
            container::point<std::string> p{};
        }
        catch(test::exceptionalized_static_assert const& esa)
        {
            throwed = true;
            std::cout << colors::on_green << colors::bold << colors::white << esa.what() << colors::reset << std::endl;
        }
        REQUIRE(throwed);
    }
}

#undef SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT
#undef static_assert

TEMPLATE_TEST_CASE("Point construction", "[point-construction]", double, float, int)
{
    using namespace scalfmm;

    SECTION("Default construction", "[default]") { container::point<TestType> p{}; }
}

// Quick fold expression
template<typename... Args>
bool all(Args... args)
{
    return (... && args);
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
