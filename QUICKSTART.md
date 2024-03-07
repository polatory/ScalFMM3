README.md 

In this quickstart, we will present the main components of the framework.
Through a simple example, we will guide the user to make its first 
`scalfmm` code. 
All of the following tutorial can be found in `examples/tutorial.c++`.
All components we discuss below are in the `scalfmm` namespace.

## First step 

Let's choose the dimension and floating point computation type `value_type`.
```c++
constexpr int dimension = 3 ; 
using namespace scalfmm; 
using value_type = double;
```



## FMM operators and interpolators

The FFM operator consists of the near-field and far-field operators. The near-field operator is utilized in direct calculation and exclusively relies on the kernel matrix of the problem at hand. On the other hand, the far field operator employs an interpolation method based on either regularly spaced points or Chebyshev points.

### The kernels

First, the user may select the kernel of the fmm. The general form of the kernel is a vector application of $R^m$ in $R^n$, so we consider the kernel as a matrix.

Let's select the most common one, the Laplacian kernel : `1/r` with `r=|x-y|`. 
The kernel matrix will drive the number of inputs `km` and outputs `kn` attached
to each particle. The kernel matrix `matrix_kernel::one_over_r` takes one input 
and produces one output, i.e `km=1` and `kn=1`. 

In scalfmm, kernel matrices may not be the same for the far-field and the near-field.
We will discuss this feature in the section addressing the fmm operators and interpolators.
For further information about matrix kernels, look for the namespace `matrix_kernel`
in the doc.

So, to prepare the correct containers, we extract the following constants:
```c++
// the far field matrix kernel
using far_kernel_matrix_type = matrix_kernels::laplace::one_over_r;
// the near field matrix kernel
using near_kernel_matrix_type = far_kernel_matrix_type;
// number of inputs and outputs.
static constexpr std::size_t nb_inputs_near{near_kernel_matrix_type::km};
static constexpr std::size_t nb_outputs_near{near_kernel_matrix_type::kn};
static constexpr std::size_t nb_inputs_far{far_kernel_matrix_type::km};
static constexpr std::size_t nb_outputs_far{far_kernel_matrix_type::kn};
```

Once we have these constants, we can work with the particle type and then the containers. 


  
#### The near-field operator

The near-field type depends only on a kernel matrix type.
```c++
    // we define a near_field from its kernel matrix
    using near_field_type = scalfmm::operators::near_field_operator<far_kernel_matrix_type>;
```
To create the near field, in addition to the particles, it is necessary to specify whether in the calculation of the interactions between particles those if are mutual or not, i.e. whether one uses Newton's third law $f_{i,j} +f_{j,i} =0$ or not.
```c++
    near_kernel_matrix_type near_mk{};
    //
    const bool mutual = true;
    near_field_type near_field(near_mk, mutual);
```
Note that it is also possible to determine whether or not Newton's third law is used after creating the near field.
```c++
    near_field_type near_field(near_mk);    // by default, mutual = true
    near_field.mutual() = false;
```

 :warning: **Warning:** For the time being, the non-mutual feature of the kernel is used only to achieve greater parallelism. In the case where source particles coincide with target particles, the interaction of the box with itself is mutual. This will be fixed in a future release.


Additionaly, it is not even necessary to create a matrix kernel object for the near field. Since, the corresponding template parameter has already been specified, a default matrix kernel object can be directly instantiated during the construction of the near field. This allows the user to create a near field object without providing any argument. However, it should be noted that a copy of the underlying matrix kernel object can always be retrieved using an accessor.

```c++
    near_field_type near_field;
    auto near_mk_bis = near_field.matrix_kernel();  // get a copy of the default matrix kernel object
```
Finally, we also provide the possibility of creating a default matrix kernel and specify as a parameter whether or not Newton's third law should be used when constructing the near-field object.

```c++
    near_field_type near_field(mutual);
```

#### The far-field operator

The first thing to do is to define the type of interpolation we are considering.
```c++
    // we choose an interpolator with a far matrix kernel for the approximation
    using interpolator_type = interpolation::interpolator< value_type
                                                         , dimension
                                                         , far_kernel_matrix_type
                                                         , options::uniform_<options::fft_>
                                                         >;
````
The interpolator type takes an option type. The option type allows to define at compile time the type of the approximation (Chebyshev points or regularly spaced points) as well as the optimization available for the M2L operator according to the type of the points (low_rank or fft ) 
The available options are
 - `dense` the kernel matrix is in dense format
 - `low_rank` the kernel matrix is compressed in low rank format $K = U V^t$  
 - `fft` a fft optimization for uniform interpolation
You can also specify both the interpolation type (chebyshev or uniform) and the optimization with the following keys `chebyshev_dense`, `chebyshev_low_rank`, `uniform_dense`,  `uniform_low_rank`, `uniform_low_fft`.

Once the interpolator is defined, we can build the type of the far-field and create it. 
```c++
    // then, we define the far field
    using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
    //
    // define the interpolator
    interpolator_type interpolator(far_mk, order, tree_height, box.width(0));
    // define the far field
    far_field_type far_field(interpolator);
```

### FMM operators
The fmm operators is composed of two operators
- the near-field operator
- the far-field operator
  
  ```c++
      using fmm_operator_type = operators::fmm_operators<near_field_type, far_field_type>;
    // construct the fmm operator 
    fmm_operator_type fmm_operator(near_field, far_field);
  ```

The near-field and far-field can have different kernels (kernels matrix). However, the inputs and outputs must be compatible (see the advanced section)

## Containers

### Particles
The first data entry point in the framework is the `particle` type.
A particle stores the following data :
- a position of type `point<value_type, dimension>`
- `km` inputs of type `value_type` ; the number of column of the kernel matrix.
- `kn` outputs of type `value_type` ; the number of row of the kernel matrix.
- some variables stored in a tuple.

An example of particle type in 2d, with one input; 3 outputs and one variable is
```c++
static constexpr std::size_t dimension{2};
constexpr nb_inputs_near  = far_kernel_matrix_type::km ;
constexpr nb_outputs_near = far_kernel_matrix_type::kn ;

using particle_type = container::particle<
    // position
    value_type, Dimension, 
    // inputs
    value_type, nb_inputs_near, 
    // outputs
    value_type, nb_outputs_near, 
    // variables
    std::size_t // for storing the index in the original container.
    >;
 ```       
The types of inputs and outputs must be identical but can be different from the positions.


We can extract the `position_type` from the particle which is a `container::point<value_type, dimension>`
type.
```
using position_type = typename particle_type::position_type;
```
To fill the particle, we proceed as follows
```c++
        particle_type p;
        // set the position
        for(auto& e: p.position())
        {
            e = random_r();
        }
        // set the input values
        for(auto& e: p.inputs())
        {
            e = random_r();
        }
        // set the output values
        for(auto& e: p.outputs())
        {
            e = value_type(0.);
        }
        // add the variables
        p.variables(idx);
```
The variables(...) method of the particle class is a variadic method. The number of arguments of the method corresponds to the number and types of the variables in the definition of the particle type. 
 
For a set of particles, we can consider either a stl structure (vectors, ...) or a specific container proposed by `scalfmm` (`particle_container`).

### Container of particles
In `scalfmm` we provide the `particle_container`, a SOA container for the particles.
```c++
using container_type = container::particle_container<particle_type>;
// allocate 100 particles.
constexpr std::size_t nb_particles{100};
container_type container(nb_particles);
```
We need to fill this container with data. To achieve that, you can use the container
interface available in the doc.

Before putting some particles in the container, we define a box of simulation.
```c++
// box of simulation 
using box_type = scalfmm::component::box<position_type>;
// width of the box
constexpr value_type box_width{2.};
// center of the box
constexpr position_type box_center{1.,1.};
// the box for the tree 
box_type box(box_with, box_center);
```
Then, we generate particles within this simulation box.
```c++
// random generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<value_type> dis(0.0, 2.0);
auto random_r = [&dis, &gen](){ return dis(gen); };

// inserting particles in the container
for(std::size_t idx = 0; idx < nb_particles; ++idx)
{
    particle_type p;
    for(auto& e: p.position())
    {
        e = random_r();
    }
    for(auto& e: p.inputs())
    {
        e = random_r();
    }
    p.variables(idx);
    container.insert_particle(idx, p);
}
```
### Vector of particles
You may prefer to use the stl vectors as the container, as the filling of the container is different from that of the scalfmm container. The code below shows how to fill it.

```c++
using container_type = std::vector<particle_type>;
// allocate 100 particles.
constexpr std::size_t nb_particles{100};
container_type container(nb_particles);

// random generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<value_type> dis(0.0, 2.0);
auto random_r = [&dis, &gen](){ return dis(gen); };

// inserting particles in the container
for(std::size_t idx = 0; idx < nb_particles; ++idx)
{
    particle_type&  p = container[idx];
    for(auto& e: p.position())
    {
        e = random_r();
    }
    for(auto& e: p.inputs())
    {
        e = random_r();
    }
    p.variables(idx);
}
```

## Tree

The tree decomposes the simulation box into leaves and cells according to the level at which we are working.

### box of simulation
Before defining a tree, we define a box of simulation. We need the center of the box and its width.
```c++
// box of simulation 
using box_type = scalfmm::component::box<position_type>;
// width of the box
constexpr value_type box_width{2.};
// center of the box
constexpr position_type box_center{1.,1.};
// the box for the tree 
box_type box(box_width, box_center);
```
### The components 
The two components of the tree are the leaves and the cells. The sheet contains the particles while the cell contains the storage needed by the approximation. The cell will contain both the so-called multipole and local approximations. In the case of the interpolation method, these underlying structures are tensors of order of the dimension. 
```c++
    // the leaf type holding the particles
    using leaf_type = component::leaf_view<particle_type>;
   // here, we extract the correct storage for the cells from the interpolation method.
    using cell_type = component::cell<typename interpolator_type::storage_type>;
```
With this two types, the tree type is defined by
```c++
    // the tree type
    using group_tree_type = component::group_tree_view<cell_type, leaf_type, box_type>;
```

To create the tree 
```c++
    // the height of the tree
    const int tree_height = 5 ;
    // order of the approximation
    const int order = 6 ;
    // the blocking parameter for the leaves and ths cells
    const int group_leaf_size = 100;
     onst int group_cell_size = 100
    //
     group_tree_type tree(tree_height, order, box, group_leaf_size, group_cell_size, container);
```
Now that the operators and the tree are created we can execute an algorithm.


## Algorithms

We proposed three algorithms
- the full direct interactions
- `sequential` the sequential fast multipole
- `task_dep` the OpenMP fast multipole algorihm

### the direct algorithm
This algorithm is mainly used for debugging purposes. It builds the complete interactions between the particles without any approximation.
It takes the container of particles and the matrix kernel involves in the interaction.
```c++
scalfmm::algorithms::full_direct(container_source, container_source, mk_near);
```


### the fast multipole algorithm
We propose two ways to call the algorithm. The first call the method `fmm`with options. The options specify the algorithm and other things like
```c++
    scalfmm::algorithms::fmm[options::_s(options::omp)](tree, fmm_operator);
```
The options are
 - `options::omp` for the OpenMP algorithm
 - `omp_timit` for the OpenMP algorithm and display the total time
 - `options::seq`for the sequential algorithm
 - `seq_timit` for the sequential algorithm and display the time per operator
 - `timit` to display the time per operator (sequential and total time

or the qualified call
```c++
    scalfmm::algorithms::omp::task_dep(tree, fmm_operator);
```

## Advanced simulation 

### Optimized $1/r$ core for potential and force calculation

The kernel to compute potential and force is the `val_grad_one_over_r` which evaluate  
$$ (\frac{1}{|| x-y||}, \frac{x-y}{|| x-y||^3})$$
The near-field operator is defined by
```c++
//     Near-field
using near_kernel_matrix_type = scalfmm::matrix_kernels::laplace::val_grad_one_over_r<dimension>;
using near_field_type = scalfmm::operators::near_field_operator<near_kernel_matrix_type>;
```




For the far field the classical way is  to consider the same kernel as for the near-field
```c++
    //
    // We consider  the same matrix kernel for the far_field
    using far_kernel_matrix_type = near_kernel_matrix_type;   
    //
    using options = scalfmm::options::uniform_<scalfmm::options::fft_>;

    using interpolation_type = scalfmm::interpolation::interpolator<value_type, 
                                          dimension, 
                                          far_kernel_matrix_type, 
                                          options>;
    using far_field_type = scalfmm::operators::far_field_operator<interpolation_type>;
```
In this case, we construct three local arrays during the `m2l` step. 

To optimize memory and computation, we can build the potential and derive it in the *l2p* step to build the forces. To do this, we consider the `one_over_r` kernel for the `far_kernel_matrix_type` and specify in the `far_field_operator` type that we will derive the potential by adding the boolean true.

```c++
using far_kernel_matrix_type = scalfmm::matrix_kernels::laplace::one_over_r;
using interpolation_type = scalfmm::interpolation::interpolator<value_type, dimension, far_kernel_matrix_type, options>;
using far_field_type = scalfmm::operators::far_field_operator<interpolation_type, true>;
```
By doing so, the kernel is scalar in the far-field and the computational cost of `m2l`, `l2l` operators is strongly decreased while that of `l2p` increases.



### Simulation with different source and target particles
If the source particles are different from the target particles in your simulation, you must define two trees, one for the sources and one for the targets. The type of the particles can be different or not. This is the case when you consider the optimized kernel, i.e., building the forces only by deriving the potential in the `l2p` operator

Consider that the `near_kernel_matrix` and the `far_kernel_matrix_type` are different.

```c++
using particle_source_type =
  scalfmm::container::particle<value_type, dimension, value_type, 1, value_type, 1>;
using particle_target_type =
  scalfmm::container::particle<value_type, dimension, value_type, 1, value_type, 3, std::size_t>;
using leaf_source_type
 = scalfmm::component::leaf_view<particle_source_type>;
using leaf_target_type = scalfmm::component::leaf_view<particle_target_type>;

using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
using tree_source_type = scalfmm::component::group_tree_view<cell_type, leaf_source_type, box_type>;
using tree_target_type = scalfmm::component::group_tree_view<cell_type, leaf_target_type, box_type>;
```

As the leaves between the two trees are different, we must indicate via the `inject` function the type of the leaves of the source tree to be able to define correctly the interaction lists of the `p2p` operator.
```c++
namespace scalfmm::meta
{
    template<>
    struct inject<scalfmm::component::group_of_particles<leaf_target_type, particle_target_type>>
    {
        using type = std::tuple<typename scalfmm::component::group_of_particles<leaf_source_type,particle_source_type>::iterator_type,
                                scalfmm::component::group_of_particles<leaf_source_type, particle_source_type>>;
    };
}   
```
The construction of interaction lists is usually performed only once if the source and target particles do not change during the simulation. The interaction list is built only on the target tree like this
```c++
    tree_target.build_interaction_lists(tree_source, separation_criterion, false);
```
where `separation_criterion` is the separation criterion between the near field and the far field. It is defined in the matrix_kernel object (usually 1). We specify by adding false, that we do not calculate the reciprocal interaction (Newton's third law).

Finally, the call to the fmm algorithm use the two trees
``` c++
scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp_timit)](tree_source, tree_target, fmm_operator, operator_to_proceed);
```

Moreover the option specify that we consider the `OpenMP` algorithm and we display the time spend in the algorithm.




## Helper functions

### Parameters
To define and read the parameters of a program we use the Parser for Command Line options (cl_parser) available on [gitlab](https://gitlab.inria.fr/compose/legacystack/cpp_tools) in the `cpp_tools`project. The full documentation is [here](https://gitlab.inria.fr/compose/legacystack/cpp_tools/-/blob/master/cl_parser/README.md).




### IO functions

We propose several functions to

- Display  a leaf  `print_leaf(leaf)`, we display symbolic information and the particles inside the leaf. 
- isplay  a cell `print_cell(cell)`, we display symbolic information and the multipole(s) and local(s) arrays inside the lcell. 
- Display a tuple (formatted display) 
```c++ 
  std::tuple<double, int> t(0.5, 3);
  io::print()
```
the output is 
```c++ 
  [0.5, 3]
```

 - Display a particle directly through the flow operator
### meta functions
Alle these functions are located in the directory *scalfmm/meta*
#### check the type of a variable at the compilation
```c++

    for(auto const p_ref: leaf)
    {
        scalfmm::meta::td<decltype(p_ref)> t;
     }
```
The error at the compilation is

```c++
 error: 'scalfmm::meta::td<const std::tuple<double&, double&, double&, double&, long unsigned int&> > t' has incomplete type
  147 |                                      meta::td<decltype(p_ref)> t;
      |                                                                ^
```
The meta function is located in scalfmm/meta/utils.hpp.

#### repeat function


  The `repeat` function applies the lambda function to each element the object. 


  Let's consider in this example that *locals_iterator* is an iterator tuple, the lambda function increments an element. The meta function will increment each element of the tuple
 ```c++
 meta::repeat([](auto& it) { ++it; }, locals_iterator);
 ```
See sphinx documentation for more examples.
#### for_each function

  The `for_each` function applies the lambda function to each element the object.
