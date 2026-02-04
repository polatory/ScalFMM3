#include "scalfmm/algorithms/fmm.hpp"
#include "scalfmm/algorithms/full_direct.hpp"
#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_impl.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/interpolation/interpolation.hpp"
#include "scalfmm/matrix_kernels/laplace.hpp"
#include "scalfmm/operators/fmm_operators.hpp"
#include "scalfmm/tools/fma_loader.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/for_each.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/leaf_view.hpp"
#include "scalfmm/utils/accurater.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "py_interface.hpp"

namespace py = pybind11;

template<typename ValueType>
using py_array_t = py::array_t<ValueType, py::array::f_style>;   // to enforce column-major storage

namespace pyfmm
{

    using value_type = double;
    static constexpr std::size_t dimension{2};

    // the far field matrix kernel
    using far_matrix_kernel_type = scalfmm::matrix_kernels::laplace::one_over_r;
    // the near field matrix kernel
    using near_matrix_kernel_type = far_matrix_kernel_type;
    // number of inputs and outputs.
    static constexpr std::size_t nb_inputs_near{near_matrix_kernel_type::km};
    static constexpr std::size_t nb_outputs_near{near_matrix_kernel_type::kn};

    // interpolator
    using interpolator_type = scalfmm::interpolation::interpolator<value_type, dimension, far_matrix_kernel_type,
                                                                   scalfmm::options::uniform_<scalfmm::options::fft_>>;

    // operators for the fmm algorithm
    using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
    using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
    using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;

    // particle
    using particle_type = scalfmm::container::particle<
      // position
      value_type, dimension,
      // inputs
      value_type, nb_inputs_near,
      // outputs
      value_type, nb_outputs_near,
      // variables
      std::size_t   // for storing the index in the original container.
      >;

    // position point type
    using position_type = typename particle_type::position_type;

    // simulation box
    using box_type = scalfmm::component::box<position_type>;

    // particle container
    using container_type = std::vector<particle_type>;

    // morton type
    using morton_type = std::size_t;

    // group tree type
    using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
    using leaf_type = scalfmm::component::leaf_view<particle_type>;
    using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

    // numpy array object
    using np_array = py_array_t<value_type>;
    using np_array_int = py_array_t<int>;

    namespace to_cpp
    {
        template<typename T>
        auto to_ptr(py_array_t<T> const& array) -> T*
        {
            py::buffer_info buf_info = array.request();
            return static_cast<T*>(buf_info.ptr);
        }
    }   // namespace to_cpp

}   // namespace pyfmm

PYBIND11_MAKE_OPAQUE(pyfmm::container_type);

PYBIND11_MAKE_OPAQUE(std::initializer_list<pyfmm::value_type>);

PYBIND11_MAKE_OPAQUE(std::vector<pyfmm::morton_type>);

PYBIND11_MODULE(pyfmm, m)
{
    m.doc() = "Fast Multipole Method python module";   // optional module docstring

    using value_type = pyfmm::value_type;
    py::bind_vector<std::vector<pyfmm::morton_type>>(m, "VecMorton");
    // --------------
    // scalfmm point
    // --------------
    using point = pyfmm::position_type;
    py::class_<point>(m, "point", "multidimensional point")
      .def(py::init<value_type, value_type>(), "main constructor")
      .def("norm", &point::norm, "return the 2-norm of the point")
      .def(
        "__getitem__", [](const point& p, int i) -> point::value_type const& { return p[i]; },
        py::return_value_policy::reference, py::is_operator())
      .def(
        "__setitem__", [](point& p, int i, value_type value) -> void { p[i] = value; },
        py::return_value_policy::reference, py::is_operator())
      .def("__repr__",
           [](const point& p) -> std::string
           {
               std::ostringstream oss;
               oss << p;
               return oss.str();
           });

    // ------------
    // scalfmm box
    // ------------
    using box = pyfmm::box_type;
    py::class_<box>(m, "box")
      .def(py::init<point, value_type>(), py::arg("min_corner"), py::arg("side_length"),
           "init box from min corner and its width")
      .def(py::init<value_type, point>(), py::arg("width"), py::arg("box_center"),
           "init box from its center and its width")
      .def(py::init<point, point>(), py::arg("corner_1"), py::arg("corner_2"), "Init box from 2 corners")
      .def("corner", static_cast<point (box::*)(std::size_t) const>(&box::corner), py::arg("idx"),
           "accessor for the box corners")
      .def("corner", static_cast<void (box::*)(std::size_t, const point&)>(&box::corner), py::arg("idx"),
           py::arg("pos"), "Setter for the corners")
      .def("width", &box::width, py::arg("dim"), "Returns the width for given dimension")
      .def("center", &box::center, "Accessor for the box center")
      .def("contains", static_cast<bool (box::*)(const point& p) const>(&box::contains), py::arg("point"),
           "checks whether a position is within the box bounds")
      .def("__repr__",
           [](const box& b) -> std::string
           {
               std::ostringstream oss;
               oss << b;
               return oss.str();
           });

    // ----------------
    // scalfmm particle
    // ----------------
    using particle = pyfmm::particle_type;
    py::class_<particle>(m, "particle")
      .def(py::init<>(), "default constructor")
      .def("__repr__",
           [](const particle& part) -> std::string
           {
               std::ostringstream oss;
               oss << part;
               return oss.str();
           })
      .def("get_position",
           [](const particle& part, std::size_t i) -> particle::base_type::position_value_type
           {
               if(i >= part.sizeof_dimension())
                   throw py::index_error();
               return part.position(i);
           })
      .def("set_position",
           [](particle& part, std::size_t i, particle::base_type::position_value_type value) -> void
           {
               if(i >= part.sizeof_dimension())
                   throw py::index_error();
               part.position(i) = value;
           })
      .def("get_input",
           [](const particle& part, std::size_t i) -> particle::base_type::inputs_value_type
           {
               if(i >= part.sizeof_inputs())
                   throw py::index_error();
               return part.inputs(i);
           })
      .def("set_input",
           [](particle& part, std::size_t i, particle::base_type::inputs_value_type value) -> void
           {
               if(i >= part.sizeof_inputs())
                   throw py::index_error();
               part.inputs(i) = value;
           })
      .def("get_output",
           [](const particle& part, std::size_t i) -> particle::base_type::outputs_value_type
           {
               if(i >= part.sizeof_outputs())
                   throw py::index_error();
               return part.outputs(i);
           })
      .def("set_output",
           [](particle& part, std::size_t i, particle::base_type::outputs_value_type value) -> void
           {
               if(i >= part.sizeof_outputs())
                   throw py::index_error();
               part.outputs(i) = value;
           })
      .def("set_variable", [](particle& part, std::size_t value) -> void { part.variables(value); })
      .def("sizeof_dimension", &particle::sizeof_dimension)
      .def("sizeof_inputs", &particle::sizeof_inputs)
      .def("sizeof_outputs", &particle::sizeof_outputs)
      .def("sizeof_variables", &particle::sizeof_variables);

    // --------------------------
    // scalfmm particle container
    // --------------------------
    using container = pyfmm::container_type;
    py::class_<container>(m, "container")
      .def(py::init<>())
      .def(py::init<std::size_t>())
      .def("clear", &container::clear)
      .def("pop_back", &container::pop_back)
      .def("__len__", [](const container& c) -> container::size_type { return c.size(); })
      .def(
        "__iter__", [](container& c) { return py::make_iterator(c.begin(), c.end()); },
        py::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */
      .def(
        "__getitem__", [](container& c, int i) -> particle& { return c.at(i); },
        py::return_value_policy::reference_internal, py::is_operator())
      .def(
        "__setitem__", [](container& c, int i, const particle& part) -> void { c.at(i) = part; }, py::is_operator())
      .def("__repr__",
           [](container& c) -> std::string
           {
               std::ostringstream os;
               for(py::ssize_t idx{0}; idx < c.size(); ++idx)
               {
                   os << c[idx] << std::endl;
               }
               return os.str();
           })
      .def("get_outputs",
           [](container& c, pyfmm::np_array& out) -> void
           {
               static constexpr py::ssize_t outputs_size = container::value_type::base_type::outputs_size;
               auto out_proxy = out.template mutable_unchecked<1>();

               const py::ssize_t out_size = out_proxy.shape(0);
               const std::size_t nb_particles{c.size()};

               if(out_size != nb_particles * outputs_size)
                   throw std::runtime_error("Wrong size for output vector");

               for(py::ssize_t idx{0}; idx < nb_particles; ++idx)
               {
                   for(py::ssize_t i{0}; i < outputs_size; ++i)
                   {
                       out_proxy(idx + i * outputs_size) = c[idx].outputs(i);
                   }
               }
           });

    // ------------
    // scalfmm tree
    // ------------
    using tree = pyfmm::group_tree_type;
    py::class_<tree>(m, "tree")

      //   .def(py::init<std::size_t, std::size_t, std::size_t, box>(), py::arg("tree_height"), py::arg("order"),
      //        py::arg("group_size_leaves"), py::arg("box"))

      .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, box>(), py::arg("tree_height"),
           py::arg("order"), py::arg("group_size_leaves"), py::arg("group_size_cells"), py::arg("box"))

      //   .def(py::init<std::size_t, std::size_t, box, std::size_t, std::vector<pyfmm::morton_type>>(),
      //        py::arg("tree_height"), py::arg("order"), py::arg("box"), py::arg("group_size_leaves"), py::arg("morton"))

      .def(py::init<std::size_t, std::size_t, box, std::size_t, std::size_t, container, bool>(), py::arg("tree_height"),
           py::arg("order"), py::arg("box"), py::arg("group_size_leaves"), py::arg("group_size_cells"),
           py::arg("particle_container"), py::arg("particles_are_sorted"))

      .def(py::init<std::size_t, std::size_t, box, std::size_t, container, bool>(), py::arg("tree_height"),
           py::arg("order"), py::arg("box"), py::arg("group_size"), py::arg("particle_container"),
           py::arg("particles_are_sorted"))

      .def("leaf_level", &tree::leaf_level, " return the leaf level")
      .def("top_level", &tree::top_level, "return the top level used in the algorithm")
      .def("box_center", &tree::box_center, "return the center of the simulation box")
      .def("box_width", &tree::box_width, "return the width of the simulation box")
      .def("leaf_width", &tree::leaf_width, "return the width of a leaf")
      .def("reset_outputs", &tree::reset_outputs, "reset all outputs")
      .def("reset_far_field", &tree::reset_far_field, "reset far field")
      .def("reset_multipoles", &tree::reset_multipoles, "reset multipoles")
      .def("reset_locals", &tree::reset_locals, "reset locals")
      //   .def("build_with_morton", &tree::construct<pyfmm::morton_type>, "build tree with morton indexes")
      .def("build_with_morton", &tree::build_with_morton, "build tree with morton indexes")
      .def("fill_leaves_with_particles", &tree::fill_leaves_with_particles<container>,
           "fill tree with particles containers")
      //   .def("get_outputs",
      //        [](tree& tree, pyfmm::np_array& out) -> void
      //        {
      //            static constexpr py::ssize_t outputs_size = container::value_type::base_type::outputs_size;
      //            auto out_proxy = out.template mutable_unchecked<1>();

      //            const py::ssize_t out_size = out_proxy.shape(0);
      //            py::ssize_t index;

      //            for(auto const pg: tree.vector_of_leaf_groups())
      //            {
      //                for(auto const& leaf: pg->components())
      //                {
      //                    for(auto const particle_tuple_ref: leaf)
      //                    {
      //                        auto p = typename tree::leaf_type::const_proxy_type(particle_tuple_ref);

      //                        const auto& idx = std::get<0>(p.variables());
      //                        auto const& output = p.outputs();

      //                        for(py::ssize_t i{0}; i < outputs_size; ++i)
      //                        {
      //                            index = idx + i * outputs_size;

      //                            if(index >= out_size)
      //                                throw std::runtime_error("index out of range");

      //                            out_proxy(index) = output.at(i);
      //                        }
      //                    }
      //                }
      //            }
      //        })
      .def("get_outputs",
           [](tree& tree, pyfmm::np_array& outputs_matrix) -> void
           {
               static constexpr py::ssize_t outputs_size = container::value_type::base_type::outputs_size;
               using inputs_type = container::value_type::base_type::outputs_type::value_type;

               py::buffer_info buf_info = outputs_matrix.request();
               auto N = buf_info.shape[0];
               auto nb_inputs = buf_info.shape[1];
               // set the numpy matrix in xtensor (column major)
               auto xt_matrix = scalfmm::python::to_cpp::to_matrix(outputs_matrix);
               //
               int idx{0};
               // loop through the groups of leaves
               for(auto const& pg: tree.vector_of_leaf_groups())
               {
                   std::array<std::add_pointer_t<inputs_type>, outputs_size> output_storage;
                   //
                   auto nb_elements_in_group = pg->storage().size();
                   // Get the storage of the inputs (of the current group)
                   output_storage[0] = pg->storage().ptr_on_output();
                   for(int i = 1; i < outputs_size; ++i)
                   {
                       output_storage[i] = output_storage[i - 1] + nb_elements_in_group;
                   }

                   // loop through the elements of the group
                   for(auto i(0); i < nb_elements_in_group; ++i, ++idx)
                   {
                       for(int k = 0; k < outputs_size; ++k)
                       {
                           xt_matrix(idx, k) = (output_storage[k])[i];
                       }
                   }
               }
           })
      .def("fill_particles",
           [](tree& tree, pyfmm::np_array& position_matrix) -> void
           {
               static constexpr py::ssize_t inputs_size = container::value_type::base_type::inputs_size;
               using position_type = container::value_type::base_type::position_type::value_type;

               py::buffer_info buf_info = position_matrix.request();
               auto N = buf_info.shape[0];
               auto dimension = buf_info.shape[1];
               // set the numpy matrix in xtensor (column major)
               auto xt_pos = scalfmm::python::to_cpp::to_matrix(position_matrix);
               //
               // loop through the groups of leaves
               std::vector<std::add_pointer_t<position_type>> ptr(dimension, nullptr);
               int idx{0};
               int cc{0};
               for(auto const& pg: tree.vector_of_leaf_groups())
               {
                   auto nb_particles = pg->storage().size();
                   auto ptr_start = pg->storage().ptr_on_position();
                   for(int j = 0; j < dimension; ++j)
                   {
                       auto ptr = ptr_start + nb_particles * j;
                       int idx1{idx};

                       for(auto i = 0; i < nb_particles; ++i, ++idx1)
                       {
                           ptr[i] = xt_pos(idx1, j);
                       }
                   }
                   idx += nb_particles;
               }
           })
      .def("set_inputs",
           [](tree& tree, pyfmm::np_array& inputs_matrix) -> void
           {
               static constexpr py::ssize_t inputs_size = container::value_type::base_type::inputs_size;
               using inputs_type = container::value_type::base_type::inputs_type::value_type;

               py::buffer_info buf_info = inputs_matrix.request();
               auto N = buf_info.shape[0];
               auto nb_inputs = buf_info.shape[1];
               // set the numpy matrix in xtensor (column major)
               auto xt_inputs = scalfmm::python::to_cpp::to_matrix(inputs_matrix);
               //
               int idx{0};
               // loop through the groups of leaves
               for(auto const& pg: tree.vector_of_leaf_groups())
               {
                   std::array<std::add_pointer_t<inputs_type>, inputs_size> input_storage;
                   //    std::array<inputs_type*, inputs_size> input_storage;
                   auto nb_elements_in_group = pg->storage().size();
                   // Get the storage of the inputs (of the current group)
                   input_storage[0] = pg->storage().ptr_on_input();
                   for(int i = 1; i < inputs_size; ++i)
                   {
                       input_storage[i] = input_storage[i - 1] + nb_elements_in_group;
                   }

                   // loop through the elements of the group
                   for(auto i(0); i < nb_elements_in_group; ++i, ++idx)
                   {
                       for(int k = 0; k < inputs_size; ++k)
                       {
                           (input_storage[k])[i] = xt_inputs(idx, k);
                       }
                   }
               }
           })

      .def("__repr__",
           [](tree& tree) -> std::string
           {
               std::stringstream os;
               os << "tree | height = " << tree.height() << std::endl;
               os << "tree | order = " << tree.order() << std::endl;
               os << "tree | group size for leaves = " << tree.group_of_leaf_size() << std::endl;
               os << "tree | group size for cells = " << tree.group_of_cell_size() << std::endl;
               py::ssize_t group_index{0};
               for(auto const pg: tree.vector_of_leaf_groups())
               {
                   os << "group n°" << group_index << std::endl;
                   for(auto const& leaf: pg->components())
                   {
                       os << "\tleaf n°" << leaf.index() << std::endl;
                       for(auto const particle_tuple_ref: leaf)
                       {
                           auto p = typename tree::leaf_type::const_proxy_type(particle_tuple_ref);
                           os << "\t\t" << p << std::endl;
                       }
                   }
                   ++group_index;
               }
               return os.str();
           });

    // -------------------------------
    // get morton index
    // -------------------------------

    m.def("morton_sort", &scalfmm::python::morton_sort<pyfmm::box_type, double, int>,
          pybind11::return_value_policy::move,
          "Get the sorted morton index and the permutation to apply on the points");
    m.def("morton_sort", &scalfmm::python::morton_sort<pyfmm::box_type, float, int>,
          pybind11::return_value_policy::move,
          "Get the sorted morton index and the permutation to apply on the points");
    //pybind11::return_value_policy::automatic,
    // ---------------------
    // scalfmm matrix kernel
    // ---------------------
    using matrix_kernel = pyfmm::far_matrix_kernel_type;
    py::class_<matrix_kernel>(m, "matrix_kernel")
      .def(py::init<>(), "defaut constructor")
      .def("name", &matrix_kernel::name, "return the name of the matrix kernel")
      .def("__repr__",
           [](matrix_kernel& mk) -> std::string
           {
               std::stringstream os;
               os << mk.name();
               return os.str();
           });

    // -------------------------------
    // scalfmm full direct computation
    // -------------------------------
    m.def(
      "full_direct", [](container& container, const matrix_kernel& matrix_kernel) -> void
      { scalfmm::algorithms::full_direct(container, matrix_kernel); }, py::arg("container"), py::arg("matrix_kernel"),
      "perform direct computation and store results in the container");

    // -----------------
    // scalfmm operators
    // -----------------
    using interpolator = pyfmm::interpolator_type;
    using far_field = pyfmm::far_field_type;
    using near_field = pyfmm::near_field_type;
    using fmm_operator = pyfmm::fmm_operator_type;

    // interpolator
    py::class_<interpolator>(m, "interpolator")
      .def(py::init<std::size_t, std::size_t, value_type, value_type>(), py::arg("order"), py::arg("tree_height"),
           py::arg("root_cell_width"), py::arg("cell_width_extension") = value_type(0.));

    // far field
    py::class_<far_field>(m, "far_field").def(py::init<const interpolator&>(), py::arg("interpolator"));

    // near field
    py::class_<near_field>(m, "near_field")
      .def(py::init<>(), "default constructor")
      .def(py::init<bool>(), py::arg("mutual"), "first constructor")
      .def(py::init<matrix_kernel>(), py::arg("matrix_kernel"), "second constructor")
      .def(py::init<matrix_kernel, bool>(), py::arg("matrix_kernel"), py::arg("mutual"), "third constructor")
      //   .def("matrix_kernel", &near_field::matrix_kernel, py::return_value_policy::reference_internal,
      //        "underlying matrix kernel accessor")
      .def_property(
        "separation_criterion", [](const near_field& nf) -> int { return nf.separation_criterion(); },
        [](near_field& nf, int value) { nf.separation_criterion() = value; })
      .def_property(
        "mutual", [](const near_field& nf) -> bool { return nf.mutual(); },
        [](near_field& nf, bool value) { nf.mutual() = value; })
      .def("__repr__",
           [](near_field& nf)
           {
               return "near field[ mk: " + nf.matrix_kernel().name() +
                      ", separation criterion: " + std::to_string(nf.separation_criterion()) +
                      ", mutual: " + std::to_string(nf.mutual()) + " ]";
           });

    // fmm operators
    py::class_<fmm_operator>(m, "fmm_operator")
      .def(py::init<const near_field&, const far_field&>(), py::arg("near_field"), py::arg("far_field"));

    // ---------------------
    // scalfmm fmm algorithm
    // ---------------------
    m.def(
      "fmm_seq", [](tree& tree, const fmm_operator& fmm_operator) -> void
      { scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](tree, fmm_operator); }, py::arg("tree"),
      py::arg("fmm_operators"), "perform the fmm algorithm and store results in the tree (sequential version)");

    m.def(
      "fmm_omp", [](tree& tree, const fmm_operator& fmm_operator) -> void
      { scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](tree, fmm_operator); }, py::arg("tree"),
      py::arg("fmm_operators"), "perform the fmm algorithm and store results in the tree (parallel version)");

    // ---------
    // accuracy
    // ---------
    m.def(
      "compute_error",
      [](container const& container, tree const& tree)
      {
          scalfmm::utils::accurater<value_type> error;

          scalfmm::component::for_each_leaf(std::cbegin(tree), std::cend(tree),
                                            [&container, &error](auto const& leaf)
                                            {
                                                // loop on the particles of the leaf
                                                for(auto const p_ref: leaf)
                                                {
                                                    // build a particle
                                                    const auto p = typename tree::leaf_type::const_proxy_type(p_ref);
                                                    //
                                                    const auto& idx = std::get<0>(p.variables());

                                                    auto const& output_ref = container[idx].outputs();
                                                    auto const& output = p.outputs();

                                                    for(int i{0}; i < pyfmm::nb_outputs_near; ++i)
                                                    {
                                                        error.add(output_ref.at(i), output.at(i));
                                                    }
                                                }
                                            });
          return py::make_tuple(error.get_relative_l2_norm(), error.get_relative_infinity_norm(),
                                error.get_rms_error());
      },
      py::arg("container"), py::arg("tree"), "compute error between direct and fmm results");

    m.def(
      "compute_error",
      [](pyfmm::np_array& out_direct, pyfmm::np_array& out_fmm)
      {
          scalfmm::utils::accurater<value_type> error;
          auto xt_direct = scalfmm::python::to_cpp::to_matrix(out_direct);
          auto xt_fmm = scalfmm::python::to_cpp::to_matrix(out_fmm);

          auto const& shape_direct = xt_direct.shape();
          auto const& shape_fmm = xt_fmm.shape();
          if(shape_direct[0] != shape_fmm[0] or shape_direct[1] != shape_fmm[1])
          {
              std::cerr << " shape(data1) (" << shape_direct[0] << ", " << shape_direct[1] << ")\n "
                        << " shape(data2) (" << shape_fmm[0] << ", " << shape_fmm[1] << ")\n ";
              throw std::runtime_error("Vectors have different size ()");
          }
          if(shape_fmm[1] != 1)
          {
              throw std::runtime_error("only output_size== 1 is available");
          }
          for(py::ssize_t i{0}; i < shape_direct[0]; ++i)
          {
              error.add(xt_direct(i, 1), xt_fmm(i, 1));
          }

          return py::make_tuple(error.get_relative_l2_norm(), error.get_relative_infinity_norm(),
                                error.get_rms_error());
      },
      py::arg("out_direct"), py::arg("out_fmm"), "compute error between direct and fmm results");

    using fmaloader = scalfmm::io::FFmaGenericLoader<value_type, pyfmm::dimension>;

    py::class_<fmaloader>(m, "fmaloader")
      .def(py::init<const std::string&, bool>(), py::arg("name"), py::arg("verbose"))

      // void
      // fillParticles(vector_type & dataToRead, const std::size_t N)
      .def("n", &fmaloader::getNumberOfParticles)
      .def("nb_data", &fmaloader::getNbRecordPerline)
      .def("data_type", &fmaloader::getDataType)
      .def("dim", &fmaloader::get_dimension)
      .def("nb_inputs", &fmaloader::get_number_of_input_per_record)
      .def("nb_outputs", &fmaloader::get_number_of_output_per_record)
      .def("data_type", &fmaloader::getDataType)
      .def("centre", &fmaloader::getBoxCenter)
      .def("width", &fmaloader::getBoxWidth)
      .def("close", &fmaloader::close)
      .def(
        "read",
        [](fmaloader& loader)
        {
            int nb_line_record = loader.getNbRecordPerline();
            int nb_particles = loader.getNumberOfParticles();
            std::vector<value_type> data(nb_particles * nb_line_record);
            loader.fillParticles(data, nb_particles);
            return py::array_t<value_type, py::array::c_style>({nb_particles, nb_line_record}, data.data());
        },
        pybind11::return_value_policy::move, " read the data (matrix C form)");

    /////////////////////////////////////////////////////////////////////////
    //
    // Writer interface
    //
    using fmawriter = scalfmm::io::FFmaGenericWriter<value_type>;

    py::class_<fmawriter>(m, "fmawriter")
      .def(py::init<const std::string&>(), py::arg("name"), " open a file to store the data. (extension fma or bfma) ")
      .def(py::init<const std::string&, bool>(), py::arg("name"), py::arg("binary"),
           " open a file to store the data. (extension fma or bfma) ")

      .def("close", &fmawriter::close, " close the file")
      .def(
        "write_header",
        [](fmawriter& writer, pyfmm::np_array centre, const value_type& width, const int nb_particles,
           const int dataType, const int nbDataPerRecord, const int dimension, const int nb_input_values)
        {
            const value_type* ptr = pyfmm::to_cpp::to_ptr(centre);
            writer.writeHeader(ptr, width, nb_particles, dataType, nbDataPerRecord, dimension, nb_input_values);
        },
        " write the header")
      .def("write", &fmawriter::writeDataFromTree<typename pyfmm::group_tree_type>, " write the data particles")
      .def(
        "write",
        [](fmawriter& writer, pyfmm::np_array particles, const int full_size, const int nb_particles)
        {
            const value_type* ptr = pyfmm::to_cpp::to_ptr(particles);
            writer.writeArrayOfReal(ptr, full_size, nb_particles);
        },
        " write the data particles");
}