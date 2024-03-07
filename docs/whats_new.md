# What's new in **ScalFMM 3.0**

- Add barycentric interpolator class
- Update of the interpolator's constructors:

```c++
using value_type = double;
static constexpr std::size_t dimension{3};

using far_matrix_kernel_type = scalmm::matrix_kernels::one_over_r;
using interpolator_type = scalfmm::interpolator<value_type, dimension, far_matrix_kernel_type, scalfmm::options::uniform_<scalfmm::options::fft_>>;

far_matrix_kernel_type mk;

interpolator_type interp1(order, tree_height, box.width(0));
interpolator_type interp2(order, tree_height, box.width(0), is_periodic);
interpolator_type interp3(order, tree_height, box.width(0), is_periodic, cell_width_extension);

interpolator_type interp4(mk, order, tree_height, box.width(0));
interpolator_type interp5(mk, order, tree_height, box.width(0), is_periodic);
interpolator_type interp6(mk, order, tree_height, box.width(0), is_periodic, cell_width_extension);

interpolator_type interp7(order, tree_height, box);
interpolator_type interp8(order, tree_height, box, cell_width_extension);

interpolator_type interp9(mk, order, tree_height, box);
interpolator_type interp10(mk, order, tree_height, box, cell_width_extension);

```

- Some bugs related to the periodic boundary conditions were fixed (in progress...)