// --------------------------------
// See LICENCE file at project root
// File : group_tree.hpp
// --------------------------------
#ifndef CATALYST_RELATED_HPP
#define CATALYST_RELATED_HPP

#include "scalfmm/meta/const_functions.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/utils/parallel_manager.hpp"
#include <catalyst.h>
#include <cmath>
#include <conduit.hpp>
#include <conduit_blueprint_mesh.h>
#include <conduit_blueprint_exports.h>
#include <conduit_cpp_to_c.hpp>
#include <cstdint>
#include <inria/tcli/tcli.hpp>
#include <string>
//#include <iostream>

// catalyst
namespace catalyst_adaptor
{
    auto initialize(std::string script) -> void
    {
        parallel_manager m;
        m.init();
        // here goes scrpits of mpi init communicator
        conduit::Node node;
        node["catalyst/scripts/script" + std::to_string(0)].set_string(script);
        node["catalyst/mpi_comm"].set(0);
        catalyst_initialize(conduit::c_node(&node));
    }

    template<typename GroupTree>
    auto execute(std::size_t step, GroupTree const& tree)
    {
        conduit::Node exec_params;

        //// add time/cycle information
        auto& state = exec_params["catalyst/state"];
        state["timestep"].set(static_cast<std::int64_t>(step));
        state["time"].set(static_cast<std::double_t>(step));

        // Add channels.
        // We only have 1 channel here. Let's name it 'grid'.
        //auto particles_channel = exec_params["catalyst/channels/particles"];
        auto& box_channel = exec_params["catalyst/channels/box"];

        // Since this example is using Conduit Mesh Blueprint to define the mesh,
        // we set the channel's type to "mesh".
        box_channel["type"].set("mesh");

        const auto n_corners = scalfmm::meta::pow(2,GroupTree::dimension);

        // now create the mesh.
        auto& box_mesh = box_channel["data"];
        box_mesh["coordsets/coords/type"].set("explicit");
        box_mesh["coordsets/coords/values/x"].set(conduit::DataType::float64(n_corners));
        box_mesh["coordsets/coords/values/y"].set(conduit::DataType::float64(n_corners));

        auto const& box = tree.box();

        conduit::float64* box_sim_x = box_mesh["coordsets/coords/values/x"].value();
        conduit::float64* box_sim_y = box_mesh["coordsets/coords/values/x"].value();

        for(std::size_t n{0}; n<n_corners; ++n)
        {
            box_sim_x[n] = scalfmm::meta::get<0>(box.corner(n));
            box_sim_y[n] = scalfmm::meta::get<1>(box.corner(n));
            std::cout << box.corner(n) << '\n';
        }

        // Next, add topology
        box_mesh["topologies/mesh/type"].set("unstructured");
        box_mesh["topologies/mesh/coordset"].set("coords");
        box_mesh["topologies/mesh/elements/shape"].set("quad");
        box_mesh["topologies/mesh/elements/connectivity"].set_int32_vector({0,1,2,3});

        catalyst_execute(conduit::c_node(&exec_params));
    }

    auto finalize() -> void
    {
        conduit::Node node;
        catalyst_finalize(conduit::c_node(&node));
    }
}

#endif // CATALYST_RELATED_HPP
