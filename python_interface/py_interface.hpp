#pragma once

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// xtensor
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>

#include <iostream>
#include <span>
//
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/sort.hpp"

namespace scalfmm::python
{
    namespace py = pybind11;

    template<typename ValueType>
    using py_array_t = py::array_t<ValueType, py::array::f_style>;   // to enforce column-major storage

    using value_type = double;
    using np_array = py_array_t<value_type>;
    //
    using np_array_64 = py_array_t<double>;
    using np_array_32 = py_array_t<float>;
    //
    using py_vector_int = py::array_t<int>;

    // using matrix_type = xt::xtensor<value_type, 2 xt::layout_type::column_major>;

    namespace to_cpp
    {
        // template<typename T>
        auto to_span(py_array_t<double> const& array) -> std::span<double>
        {
            // auto a = array.unchecked<1>();
            py::buffer_info buf_info = array.request();

            return std::span<double>{static_cast<double*>(buf_info.ptr), static_cast<std::size_t>(buf_info.shape[0])};
        }
        template<typename T>
        auto to_ptr(py_array_t<T> const& array) -> T*
        {
            py::buffer_info buf_info = array.request();
            return static_cast<T*>(buf_info.ptr);
        }
        auto to_vector(py_array_t<double> const& array) -> xt::xtensor_pointer<double, 1>
        {
            // auto a = array.unchecked<1>();

            py::buffer_info buf_info = array.request();
            std::array<size_t, 1> shape{static_cast<std::size_t>(buf_info.shape[0])};
            // xt::xtensor_pointer<double, 2> a =
            return xt::adapt(static_cast<double*>(buf_info.ptr), static_cast<std::size_t>(buf_info.shape[0]),
                             xt::no_ownership(), shape);
        }
        auto matrix_to_vector(py_array_t<double> const& array) -> xt::xtensor_pointer<double, 1>
        {
            // auto a = array.unchecked<1>();

            py::buffer_info buf_info = array.request();
            std::array<size_t, 1> shape{static_cast<std::size_t>(buf_info.shape[0] * buf_info.shape[1])};
            // xt::xtensor_pointer<double, 2> a =
            return xt::adapt(static_cast<double*>(buf_info.ptr),
                             static_cast<std::size_t>(buf_info.shape[0] * buf_info.shape[1]), xt::no_ownership(),
                             shape);
        }
        auto to_matrix(py_array_t<double> const& array)   // -> xt::xtensor_pointer<double, 2>
        {
            // auto a = array.unchecked<1>();

            py::buffer_info buf_info = array.request();
            std::array<size_t, 2> shape = {static_cast<std::size_t>(buf_info.shape[0]),
                                           static_cast<std::size_t>(buf_info.shape[1])};
            // xt::xtensor_pointer<double, 2> a =
            return xt::adapt<xt::layout_type::column_major>(static_cast<double*>(buf_info.ptr), shape[0] * shape[1],
                                                            xt::no_ownership(), shape);
        }
    }   // namespace to_cpp
    /// @brief construct the Morton index and the permutation
    ///
    /// @param[in] box the simulation box containing the particles
    /// @param[in] pos the Numpy array Nxdimension containing the particles
    /// @param[in] level to sort the particles (leaf level)
    ///
    /// @return the morton index sorted ans the permutation to apply to the particles
    ///
    template<typename BoxType, typename value_type, typename Int_t>
    auto morton_sort(BoxType const& box, py_array_t<value_type> const& pos, const Int_t level)
    {
        py::buffer_info buf_info = pos.request();
        Int_t N = buf_info.shape[0];
        auto dim = buf_info.shape[1];
        // auto pos_proxy = pos.template unchecked<2>();
        // const value_type* pos_ptr = pos_proxy.data(0);
        std::vector<Int_t> permutation(N);
        std::vector<std::size_t> morton(N);

        // set the numpy matrix in xtensor (column major)
        auto xt_pos = to_cpp::to_matrix(pos);
        // build the morton intex and the permutation
        utils::morton_sort(box, xt_pos, permutation, morton, std::size_t(level));

        return std::make_tuple(morton, py_array_t<Int_t>({static_cast<Int_t>(N)}, permutation.data()));
        //    py_array_t<Int_t>({static_cast<Int_t>(N)}, permutation.data()));
    }
    void morton2(int i) { std::cout << " coucou py::morton2\n"; }

    void add(int i, int j) { std::cout << i + j; }

}   // namespace scalfmm::python