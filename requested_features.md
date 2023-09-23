# Requested Features

## Notenook

start a notebook to explain the problem, the choices, the classes, the methods,  the algorithms,  ... of what you do so that you can share approaches and come back to them more easily.

Put experiments and results in it so that you can see if it works and can easily come back to it.

## `for_each`group 
This will allow us to indroduce parallelism at group level.
Something like this
'''
    for_each_blocOfLeaves(begin, tree, []....) {
      //  here a mechanism to have a loop on the cells of the block 
    }
'''
Begin the documentation on the tree to explain the traversal mechanisms of the cells, leaves, ... with examples.
The way to traverse the leaves or cells should be first to traverse the groups
then the leaves or cells in a group. The classical way is
'''
  auto group_of_leaf_begin = std::get<0>(tree.begin());
  auto group_of_leaf_end = std::get<0>(tree.end());
  // Loop on the groups
  while (group_of_leaf_begin != group_of_leaf_end) {
     auto const& group_symbolics = (*group_of_leaf_begin)->csymbolics();

    // Loop on the leaves of the current group
    for (std::size_t leaf_index = 0;
               leaf_index < (*group_of_leaf_begin)->size(); ++leaf_index) {
      // the current leaf 
      auto& leaf = (*group_of_leaf_begin)->component(leaf_index);


    }
    ++group_of_leaf_begin;
  }
'''


## FFT optimisation
Decrease memory consumption of ffts in the transfert pass  and increase FFTW
wrapper performance. We need to introduce a generic mechanism for optimation in
the generic transfert pass. This mechanisme is not needed for Chebyshev
interpolation.
'''
 Write the algorithm
'''
Before, we should remove the list contsruction in order to have a simpler generic
code.

## FMM operator : split near field and far field 
Make an FMM operator generic enough to compose matrix kernels
What is an near or far field operators 
  - near-field operator contains
      - Kernel matrix to build/use operators P2P
      - P2P operators -- we have to specify if we construct th iteratio kernel
        at each step (loop on the particles and apply the kernel on each
        particle) or if we precompute all interaction and consider a matrix
        vecteur product.
  - far-field operator has
      - The method of approximation (expansion or interpolation). if
        interpolation methods we also need the  matrix kernel.
        For interpolation
           -  method for interpolation (points, lagrange functions (f_j(x_i) =
              \delta_{i,j}))
           - kernel function
      - the operators (P2M/M2M/M2L/L2L/L2P)

## Direct pass and others

### Interaction list and indexed access on leafs & cells
Introduce a new mechanism on the tree to retrieve an interaction lists (P2P and M2L) and build it when needed.
Indexed access will allow fast access to components.
Have a look to the adaptive tree to prepare further needed modifications.


## Accuracy for Laplace kernels

