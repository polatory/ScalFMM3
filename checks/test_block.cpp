#include <utility>
#include <array>
#include <xtensor/xtensor.hpp>
#include <scalfmm/container/block.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/tree/group_of_views.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <scalfmm/tree/utils.hpp>

template<typename B>
auto print(B const& b_) -> void
{
    const auto nb_blocks{b_.size()};
    const auto raw_size{b_.raw_size()};
    const auto ptr{b_.data()};
    std::cout << "sizeof_header : " << sizeof(typename B::header_type) << '\n';
    std::cout << "sizeof_particle : " << sizeof(typename B::particle_type) << '\n';
    std::cout << "sizeof_sym : " << sizeof(typename B::symbolics_type) << '\n';
    std::cout << "nb_blocks : " << nb_blocks << '\n';
    std::cout << "raw_size : " << raw_size << '\n';
    std::cout << "ptr : " << ptr << '\n';
}


auto main() -> int
{
    using namespace scalfmm;

    using particle_type = container::particle<float, 2, float, 3, float, 3, std::size_t>;
    using proxy_particle_type = typename particle_type::proxy_type; //container::particle<float&, 2, float&, 3, float&, 3, std::size_t&>;
    using const_proxy_particle_type = typename particle_type::const_proxy_type;//container::particle<float&, 2, float&, 3, float&, 3, std::size_t&>;
    using leaf_type = component::leaf_view<particle_type>;
    using group_type = component::group_of_particles<leaf_type, particle_type>;
    //using particle_block_type = container::particles_block<float, particle_type>;

    group_type g1_(0,5,4,8,144,true);

    print(g1_.storage());
    std::cout << "starting_index : " << g1_.csymbolics().starting_index << '\n';
    std::cout << "ending_index : " << g1_.csymbolics().ending_index << '\n';
    std::cout << "number_of_component_in_group : " << g1_.csymbolics().number_of_component_in_group << '\n';
    std::cout << "number_of_particles_in_group : " << g1_.csymbolics().number_of_particles_in_group << '\n';
    std::cout << "idx_global : " << g1_.csymbolics().idx_global << '\n';
    std::cout << "is_mine : " << g1_.csymbolics().is_mine << '\n';
    std::cout << "--------\n";

    auto& ref_sym = g1_.symbolics();
    ref_sym.idx_global = 100;
    std::cout << "idx_global : " << g1_.csymbolics().idx_global << '\n';

    auto& leaves_ = g1_.components();
    auto& particles_storage = g1_.storage();
    std::size_t leaf_index_in_group{0};
    std::size_t cnt_particles{0};

    for(auto& leaf_view: leaves_)
    {
        std::cout << "-----\n";
        auto leaf_sym_ptr = &particles_storage.symbolics(leaf_index_in_group);
        leaf_view =
          component::leaf_view<particle_type>(std::make_pair(particles_storage.begin() + cnt_particles, particles_storage.begin() + cnt_particles + 2),
                    leaf_sym_ptr);
        cnt_particles += 2;
        // accumulate for set the number of particle in group.
        const auto morton_index_of_leaf = leaf_index_in_group;
        leaf_view.index() = morton_index_of_leaf;
        // get the coordinate of the leaf in the tree
        auto coordinate = index::get_coordinate_from_morton_index<2>(morton_index_of_leaf);
        // get the corresponding box to the leaf
        ++leaf_index_in_group;
    }


    for(auto const& leaf_view: leaves_)
    {
        std::cout << std::distance(leaf_view.particles().first, leaf_view.particles().second) << '\n';
        std::cout << "-----\n";
        std::for_each(leaf_view.particles().first, leaf_view.particles().second,
                      [](auto e)
                      {
                          std::cout << "-\n";
                          //e = std::make_tuple(0.,0.,0.,0.,0.,0.,0.,0.,0);
                          proxy_particle_type(e).position() = container::point<float, 2>{1., 2.};
                          proxy_particle_type(e).inputs(0) = 11.;
                          proxy_particle_type(e).inputs(1) = 12.;
                          proxy_particle_type(e).inputs(2) = 13.;
                          proxy_particle_type(e).outputs(0) = 0.1;
                          proxy_particle_type(e).outputs(1) = 0.1;
                          proxy_particle_type(e).outputs(2) = 0.1;
                          std::get<0>(proxy_particle_type(e).variables()) = 1;
                      });
        std::for_each(leaf_view.cparticles().first, leaf_view.cparticles().second,
                  [](auto const& e) { std::cout << const_proxy_particle_type(e) << '\n'; });
    }

    return 0;
}
