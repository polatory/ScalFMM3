//
// Units for containers
// --------------------
#include <cmath>
#include <type_traits>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <string>

//#define SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT
//#include <scalfmm/utils/static_assert_as_exception.hpp>
// Followed by file where assertions must be tested

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/meta/utils.hpp"
#include <array>
#include <tuple>

// TEST_CASE("Static test", "[static-assertion]")
//{
//    using namespace scalfmm;
//
//    SECTION("Assertion test", "[assertion]")
//    {
//        bool throwed = false;
//        try
//        {
//            container::point<std::string> p{};
//        }
//        catch(test::exceptionalized_static_assert const& esa)
//        {
//            throwed = true;
//            std::cout << colors::on_green << colors::bold << colors::white << esa.what() << colors::reset <<
//            std::endl;
//        }
//        REQUIRE(throwed);
//    }
//}
//
//#undef SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT
//#undef static_assert

TEST_CASE("particle-construction", "[particle-construction]")
{
    using namespace scalfmm;

    SECTION("default-construction", "[default-construction]")
    {
        container::particle<double, 3, double, 2, double, 2, std::size_t> p{};
    }

    SECTION("proxy-construction", "[proxy-construction]")
    {
        container::particle_container<container::particle<float, 3, float, 2, float, 2, std::size_t>> c(1);
        auto it = std::begin(c);
        *it = std::make_tuple(1., 2., 3., 4., 5., 6., 7., 150);
        container::particle<float&, 3, float&, 2, float&, 2, std::size_t&> p(*it);
        auto tuple_ref = p.as_tuple();
        meta::get<0>(tuple_ref) = 0.25;
        std::cout << meta::get<0>(*it) << '\n';

        *it = std::make_tuple(1., 2., 3., 4., 5., 6., 7., 150);
        auto itc = std::cbegin(c);
        container::particle<const float&, 3, const float&, 2, const float&, 2, const std::size_t&> pc(*itc);
        auto tuple_ref_c = pc.as_tuple();
        // meta::get<0>(tuple_ref_c) = 0.25;
        std::cout << meta::get<0>(tuple_ref_c) << '\n';
        std::cout << pc << '\n';
    }
}

TEST_CASE("particle-common", "[particle-common]")
{
    using namespace scalfmm;
    using particle_type = container::particle<float, 3, float, 2, float, 2, std::size_t, std::size_t>;
    using proxy_type = container::particle<float&, 3, float&, 2, float&, 2, std::size_t&, std::size_t&>;

    SECTION("particle-common", "[particle-common]")
    {
        container::particle_container<container::particle<float, 3, float, 2, float, 2, std::size_t, std::size_t>> c(1);
        auto it = std::begin(c);
        *it = std::make_tuple(1., 1., 1., 2., 2., 3., 3., 4, 4);
        particle_type ref(*it);
        proxy_type proxy(*it);

        REQUIRE(ref.sizeof_dimension() == 3);
        REQUIRE(ref.sizeof_inputs() == 2);
        REQUIRE(ref.sizeof_outputs() == 2);
        REQUIRE(ref.sizeof_variables() == 2);
        REQUIRE(proxy.sizeof_dimension() == 3);
        REQUIRE(proxy.sizeof_inputs() == 2);
        REQUIRE(proxy.sizeof_outputs() == 2);
        REQUIRE(proxy.sizeof_variables() == 2);

        std::size_t i{0};
        meta::repeat(
          [&i, &ref, &proxy](auto const& e) {
              REQUIRE(ref.position(i) == e);
              REQUIRE(proxy.position(i) == e);
              ++i;
          },
          meta::sub_tuple(*it, typename particle_type::range_position_type{}));
        i = 0;
        meta::repeat(
          [&i, &ref, &proxy](auto const& e) {
              REQUIRE(ref.inputs(i) == e);
              REQUIRE(proxy.inputs(i) == e);
              ++i;
          },
          meta::sub_tuple(*it, typename particle_type::range_inputs_type{}));
        i = 0;
        meta::repeat(
          [&i, &ref, &proxy](auto const& e) {
              REQUIRE(ref.outputs(i) == e);
              REQUIRE(proxy.outputs(i) == e);
              ++i;
          },
          meta::sub_tuple(*it, typename particle_type::range_outputs_type{}));
        meta::repeat(
          [](auto const& e1, auto const& e2, auto const& e3) {
              REQUIRE(e1 == e2);
              REQUIRE(e1 == e3);
          },
          meta::sub_tuple(*it, typename particle_type::range_variables_type{}), ref.variables(), proxy.variables());
    }
}

TEST_CASE("particle-proxy", "[particle-proxy]")
{
    using namespace scalfmm;
    using proxy_type = container::particle<float&, 3, float&, 2, float&, 2, std::size_t&, std::size_t&>;

    SECTION("particle-proxy", "[particle-proxy]")
    {
        container::particle_container<container::particle<float, 3, float, 2, float, 2, std::size_t, std::size_t>> c(1);
        auto it = std::begin(c);
        *it = std::make_tuple(1., 1., 1., 2., 2., 3., 3., 4, 4);
        proxy_type proxy(*it);

        std::size_t i{0};
        meta::repeat(
          [&i, &proxy](auto const& e) {
              proxy.position(i) = 11.0;
              REQUIRE(11.0 == e);
              ++i;
          },
          meta::sub_tuple(*it, typename proxy_type::range_position_type{}));
        i = 0;
        meta::repeat(
          [&i, &proxy](auto const& e) {
              proxy.inputs(i) = 22.0;
              REQUIRE(22.0 == e);
              ++i;
          },
          meta::sub_tuple(*it, typename proxy_type::range_inputs_type{}));
        i = 0;
        meta::repeat(
          [&i, &proxy](auto const& e) {
              proxy.outputs(i) = 33.0;
              REQUIRE(33.0 == e);
              ++i;
          },
          meta::sub_tuple(*it, typename proxy_type::range_outputs_type{}));
        meta::repeat(
          [](auto const& e1, auto& e2) {
              e2 = 44.0;
              REQUIRE(44.0 == e1);
          },
          meta::sub_tuple(*it, typename proxy_type::range_variables_type{}), proxy.variables());
    }
}

TEST_CASE("reset-outputs", "[reset-outputs]")
{
    using namespace scalfmm;
    using particle_type = container::particle<float, 3, float, 2, float, 2, std::size_t, std::size_t>;
    using proxy_type = container::particle<float&, 3, float&, 2, float&, 2, std::size_t&, std::size_t&>;

    SECTION("reset-outputs", "[reset-outputs]")
    {
        container::particle_container<particle_type> c(10);

        auto it = std::begin(c);
        for(std::size_t i{0}; i<c.size(); ++i)
        {
            auto proxy = proxy_type(*it);
            proxy.position() = container::point<float,3>{float(i), float(i), float(i)};

            for(std::size_t ii{0}; ii<proxy.sizeof_inputs(); ++ii)
            {
                proxy.inputs(ii) = float(i);
            }
            for(std::size_t ii{0}; ii<proxy.sizeof_outputs(); ++ii)
            {
                proxy.outputs(ii) = float(i);
            }
            meta::get<0>(proxy.variables()) = i;
            meta::get<1>(proxy.variables()) = i;
            ++it;
        }

        it = std::begin(c);
        for(std::size_t i{0}; i<c.size(); ++i)
        {
            meta::repeat(
              [i](auto const& e) {
                  using type = std::decay_t<decltype(e)>;
                  std::cout << e << ' ';
                  REQUIRE(e == type(i));
              },
              *it);
            std::cout << '\n';
            ++it;
        }

        it = std::begin(c);
        for(std::size_t i{0}; i<c.size(); ++i)
        {
            auto proxy = proxy_type(*it);
            for(std::size_t ii{0}; ii<proxy.sizeof_outputs(); ++ii)
            {
                proxy.outputs(ii) = 0.;
            }
            ++it;
        }
        it = std::begin(c);
        for(std::size_t i{0}; i<c.size(); ++i)
        {
            meta::repeat(
              [i](auto const& e) {
                  using type = std::decay_t<decltype(e)>;
                  std::cout << e << ' ';
                  REQUIRE(e == 0.);
              },
              meta::sub_tuple(*it, typename proxy_type::range_outputs_type{}));
            std::cout << '\n';
            ++it;
        }

    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
