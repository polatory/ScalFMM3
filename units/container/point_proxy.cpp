//
// Units for containers
// --------------------
#include "scalfmm/container/particle_container.hpp"
#include <cmath>
#include <functional>
#include <type_traits>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <string>
#include <tuple>
#include <array>
#include <valarray>
#include "scalfmm/meta/utils.hpp"
#include <cpp_tools/colors/colorized.hpp>

//#define SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT
//#include <scalfmm/utils/static_assert_as_exception.hpp>
//// Followed by file where assertions must be tested
//#include "scalfmm/container/point.hpp"
//
//TEST_CASE("Static test", "[static-assertion]")
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
//            std::cout << colors::on_green << colors::bold << colors::white << esa.what() << colors::reset << std::endl;
//        }
//        REQUIRE(throwed);
//    }
//}
//
//#undef SCALFMM_TEST_EXCEPTIONALIZE_STATIC_ASSERT
//#undef static_assert

TEST_CASE("Point proxy construction", "[particle-construction]")
{
    using namespace scalfmm;

    SECTION("Default construction", "[default]")
    {
        std::array<float,3> t{0., 1., 2.};
        container::point<float&, 3> proxy(t);
        std::cout << t.at(0) << '\n';
        proxy.at(0) = 63.0f;
        std::cout << t.at(0) << '\n';


        // ______________________
        // TODO:
        //container::point<float, 3> p{};
        //container::point<float&, 3> p_ref(p);
        //auto mon_point_ref = p.ref();
        //auto particle_ref = container::particle_reference(*it);
        //auto point_ref = container::point_reference(*it);

        //auto position_ref = container::position(*it);
        //auto outputs_ref = container::outputs(*it);
        //
        //for(std::size_t i{0}; auto& p : outputs)
        //{
        //    p = ...;
        //}
        //
        //for(std::size_t i = 1 ...: outputs)
        //{
        //    output_ref[i] = ...;
        //}
        //construct point from range of iterator______________________
        //

        std::tuple<float, float, float> tu(0.1, 1.1, 2.1);
        container::point<float&> proxy_tu(tu);
        std::cout << meta::get<1>(tu) << '\n';
        proxy_tu.at(1) = 63.1f;
        std::cout << meta::get<1>(tu) << '\n';

        container::particle_container<container::particle<float, 3, float, 0, float, 0>> c(1);
        auto it = std::begin(c);
        *it = std::make_tuple(5.6,6.6,7.6);
        container::point<float&> proxy_it(*it);
        std::cout << meta::get<1>(*it) << '\n';
        proxy_it[1] = 63.1f;
        std::cout << meta::get<1>(*it) << '\n';
        std::cout << proxy_it << '\n';
        meta::get<1>(proxy_it) = 0.258;
        std::cout << proxy_it << '\n';

        auto itc = std::cbegin(c);
        *it = std::make_tuple(5.6,6.6,7.6);
        container::point<const float&> proxy_itc(*itc);
        std::cout << meta::get<1>(*itc) << '\n';
        //proxy_itc.at(1) = 63.1f;
        std::cout << meta::get<1>(*itc) << '\n';

        std::cout << "----------\n";
        std::cout << proxy_it << '\n';
        std::cout << proxy_tu << '\n';
        proxy_it += proxy_tu;
        std::cout << proxy_it.norm() << '\n';
    }
}

// Comparison functions
template<typename T, typename U, std::size_t D>
bool against(scalfmm::container::point<T,D> const& p1, scalfmm::container::point<U,D> const& p2)
{
    bool ok{true};
    for(std::size_t i{0}; i<D; ++i)
    {
        ok &= (p1.at(i) == p2.at(i));
    }
    return ok;
}

template<typename T, typename U, std::size_t D>
bool against(std::array<T,D> const& p1, scalfmm::container::point<U,D> const& p2)
{
    bool ok{true};
    for(std::size_t i{0}; i<D; ++i)
    {
        ok &= (p1.at(i) == p2.at(i));
    }
    return ok;
}

template<typename T, typename U, std::size_t D>
bool against(std::valarray<T> const& p1, scalfmm::container::point<U,D> const& p2)
{
    bool ok{true};
    for(std::size_t i{0}; i<D; ++i)
    {
        ok &= (p1[i] == p2.at(i));
    }
    return ok;
}

template<typename T, std::size_t D>
bool against(std::valarray<T> const& p1, std::array<T,D> const& p2)
{
    bool ok{true};
    for(std::size_t i{0}; i<D; ++i)
    {
        std::cout << p1[i] << ' ' << p2.at(i) << '\n';
        ok &= ( p1[i] == p2.at(i));
    }
    return ok;
}

template<typename U, typename... Ts, std::size_t D>
bool against(std::tuple<Ts...> const& p1, scalfmm::container::point<U,D> const& p2)
{
    bool ok{true};
    scalfmm::meta::repeat([&ok](auto const& e_p1, auto const& e_p2) { ok &= (e_p1 == e_p2); }, p1, p2);
    return ok;
}

//applu unary operator and compare the results agaisnt valarray
template<typename ValueType, typename Operator>
void apply_operator_unary(Operator&& f)
{
    using value_type = ValueType;
    std::array<float, 3> a {1., 2., 3.};
    std::array<float, 3> b {1., 2., 3.};
    std::valarray<float> a_{1., 2., 3.};
    std::valarray<float> b_{1., 2., 3.};
    value_type p_a(a);
    value_type p_b(b);
    std::invoke(std::forward<Operator>(f), p_a, p_b);
    std::invoke(std::forward<Operator>(f), a_, b_);
    REQUIRE(against(a_, p_a));
    // if TestType is a proxy
    if constexpr (std::is_same_v<value_type, scalfmm::container::point<float&>>)
    {
        REQUIRE(against(a, p_a));
    }
}

//applu binary operator and compare the results agaisnt valarray
template<typename ValueType, typename Operator>
void apply_operator_binary(Operator&& f)
{
    using value_type = ValueType;
    std::array<float, 3> a {1., 2., 3.};
    std::array<float, 3> b {1., 2., 3.};
    std::array<float, 3> c {0., 0., 0.};
    std::valarray<float> a_{1., 2., 3.};
    std::valarray<float> b_{1., 2., 3.};
    value_type p_a(a);
    value_type p_b(b);
    std::valarray<float> res_{};
    std::invoke(std::forward<Operator>(f), a_, b_, res_);
    // if TestType is a proxy
    if constexpr (std::is_same_v<value_type, scalfmm::container::point<float&>>)
    {
        value_type res_proxy(c);
        std::invoke(std::forward<Operator>(f), p_a, p_b, res_proxy);
        REQUIRE(against(res_, c));
    }
    else
    {
        value_type res{};
        std::invoke(std::forward<Operator>(f), p_a, p_b, res);
        REQUIRE(against(res_, res));
    }
}

//template test case for point and point proxy
TEMPLATE_TEST_CASE("point-construction-common", "[point-construction-common]", scalfmm::container::point<float>,
                   scalfmm::container::point<float&>)
{
    using namespace scalfmm;
    using value_type = TestType;

    SECTION("from-array", "[from-array]")
    {
        std::array<float,3> t{0., 1., 2.};
        value_type p(t);
        REQUIRE(against(t, p));
    }
    SECTION("arithmetic-operator", "[arithmetic-operator]")
    {
        apply_operator_unary<value_type>([](auto& a, auto const& b){ a += b; });
        apply_operator_unary<value_type>([](auto& a, auto const& b){ a -= b; });
        apply_operator_unary<value_type>([](auto& a, auto const& b){ a *= b; });
        apply_operator_unary<value_type>([](auto& a, auto const& b){ a /= b; });
        apply_operator_unary<value_type>([](auto& a, [[maybe_unused]] auto const& b){ a += float(8.); });
        apply_operator_unary<value_type>([](auto& a, [[maybe_unused]] auto const& b){ a -= float(8.); });
        apply_operator_unary<value_type>([](auto& a, [[maybe_unused]] auto const& b){ a *= float(8.); });
        apply_operator_unary<value_type>([](auto& a, [[maybe_unused]] auto const& b){ a /= float(8.); });
        apply_operator_binary<value_type>([](auto const& a, auto const& b, auto& r){ r =  a + b; });
        apply_operator_binary<value_type>([](auto const& a, auto const& b, auto& r){ r =  a - b; });
        apply_operator_binary<value_type>([](auto const& a, auto const& b, auto& r){ r =  a * b; });
        apply_operator_binary<value_type>([](auto const& a, auto const& b, auto& r){ r =  a / b; });
        apply_operator_binary<value_type>([](auto const& a, [[maybe_unused]] auto const& b, auto& r){ r = a + float(8.); });
        apply_operator_binary<value_type>([](auto const& a, [[maybe_unused]] auto const& b, auto& r){ r = a - float(8.); });
        apply_operator_binary<value_type>([](auto const& a, [[maybe_unused]] auto const& b, auto& r){ r = a * float(8.); });
        apply_operator_binary<value_type>([](auto const& a, [[maybe_unused]] auto const& b, auto& r){ r = a / float(8.); });
        apply_operator_binary<value_type>([](auto const& a, [[maybe_unused]] auto const& b, auto& r){ r = a + float(8.); });
        apply_operator_binary<value_type>([](auto const& a, [[maybe_unused]] auto const& b, auto& r){ r = float(8.) * a; });
    }
}

int main(int argc, char* argv[])
{
    // global setup...
    int result = Catch::Session().run(argc, argv);
    // global clean-up...
    return result;
}
