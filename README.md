# ScalFMM: Fast Multipole Method

[![pipeline status](https://gitlab.inria.fr/solverstack/ScalFMM/badges/develop/pipeline.svg)](https://gitlab.inria.fr/solverstack/ScalFMM/commits/experimental)
[![coverage report](https://gitlab.inria.fr/solverstack/ScalFMM/badges/develop/coverage.svg)](https://gitlab.inria.fr/solverstack/ScalFMM/commits/experimental)


**ScalFMM** is a C++ library that implements a kernel independent Fast Multipole Method.


Copyright Inria, please read the licence.

## Requirements

  - CMake v3.18.0 or later
  - C++ compiler that supports
    - C++17 [compiler support list](http://en.cppreference.com/w/cpp/compiler_support)
    - [OpenMP](http://www.openmp.org/resources/openmp-compilers/)
  - Custom BLAS, FFT implementations.

The following are optional:

  - [Doxygen](https://www.doxygen.nl/index.html) to build the documentation.
  - [Sphinx](https://www.sphinx-doc.org/en/master/) and extensions (see the doc section) to build the documentation. In this case, we need the following packages: breathe, exhale, recommonmark and sphinx-rtd-theme.
  - 
  - An MPI implementation to build the distributed files.
  - [StarPU](http://starpu.gforge.inria.fr/) for the relevant FMM implementations.

ScalFMM depends on the complete [`xtensor`](https://github.com/xtensor-stack) stack. The following modules are embedded as submodules :
      - xtl
      - xsimd
      - xtensor
      - xtensor-blas
      - xtensor-fftw

We also retrieve [`cpp_tools`](https://gitlab.inria.fr/compose/legacystack/cpp_tools) as a submodule which is a collection of small cpp helpers.


## Get and Build ScalFMM

### Cloning

To use last development states of ScalFMM, please clone the master
  branch. Note that ScalFMM contains multiple git submodules like `morse_cmake` and the `xtensor` software stack.
  To get sources please use these commands:
``` bash
git clone --recursive https://gitlab.inria.fr/solverstack/ScalFMM.git -b experimental
```
or
```bash
git clone https://gitlab.inria.fr/solverstack/ScalFMM.git -b experimental
cd ScalFMM
git submodule init
git submodule update

``` 
### Building
The project has been tested  with the following compiler
  - GCC 11, 12  both on Linux and OSX
  - Clang/LLVM 14.0, 16.0 

ScalFMM uses iterators on task dependencies (OpenMP 5.x) the following compilers do not support them yet
  - LLVM 13.x, 15.x 
  - Intel OneAPI 2022 and lower

To keep a clean source tree, do an out-of-source build by creating a `build` folder out of your clone.
#### OpenMP or sequential installation

``` bash
cd /path/to/build/
# Use cmake, with relevant options
cmake .. # -Dscalfmm_USE_MKL=ON 
```

The build may be configured after the first CMake invocation using, for instance, `ccmake` or `cmake-gui`.

```bash
# Still in the build folder
ccmake .
# Or
cmake-gui .
```

#### Optimization

Customization or Optimized options:
  - If you plan to use Intel MKL, please set `scalfmm_USE_MKL` to `ON`. This will also choose MKL for the FFTs and prevent unusual behavior with multithreaded MKL version. 
  - if we are interested by periodic boundary condition, the option `scalfmm_BUILD_PBC` have to be set to `ON`. 
  - 
The binaries are then compiled calling `make` (or `ninja` if you specified it at the configure step).

The `all` target only build the tools binaries. These are binaries that can generate particle distributions, 
perform direct fmm computation, etc...

Invoke `make help` to see the available targets.
Gloabal targets are available :
* `examples` builds the examples in the `examples` folder,
* `units` builds the unit tests in the `units` folder.


If you want the examples, run `make examples`.

You can also specify your install directory with `-DCMAKE_INSTALL_PREFIX=/path/to/your/install` and then
call `make install`

#### MPI installation

##### On macosX
 Be careful if you install openMPI with brew, you must compile with the gcc version of brew. To do construct openmpi with gcc@12, we run the following command

```bash
brew install open-mpi  --cc=gcc-12
```
and make be sure that `mpicc`, `mpic++` and `mpiexec` are in your path.

Then to construct `ScalFMM` with MPI do
```bash
 CC=/usr/local/Cellar/open-mpi/4.1.5/bin/mpicc \
 CXX=/usr/local/Cellar/open-mpi/4.1.5/bin/mpic++ \
 cmake -DCMAKE_CXX_FLAGS="-march=native" -Dscalfmm_USE_MKL=ON  -Dscalfmm_USE_MPI=ON  -S ../ -B.
```

### Performance

When building ScalFMM, remember that performance depends on SIMD support.
If you target a specific architecture, please, add the corresponding SIMD flag to your release flags.
For example, if you are targeting an AVX2 enabled processor, add `-mavx2` to your flags.
And if you know the microarchitecture, like cascadelake for example, you can also add the flag `-march=cascadelake`.
When adding arch flags, SIMD flags will be automaticaly triggered. 

## Using ScalFMM in your project

To find ScalFMM, `pkgconfig` can be used within your CMake and all ScalFMM dependencies will be found automatically.
Here is an example :

```cmake
find_package(scalfmm CONFIG REQUIRED)
if(scalfmm_FOUND)
  message(STATUS "ScalFMM Found")
  add_executable(my_exe program.cpp )
  target_link_libraries(my_exe scalfmm::scalfmm)
else()
  message(FATAL_ERROR "ScalFMM NOT FOUND")
endif()
```

`pkgconfig` will find the following targets:
- `scalfmm::scalfmm` with sequential and OpenMP support.
- `scalfmm::scalfmm-headers` with only the headers, requires full configuration and linking with dependencies on your side.
- `scalfmm::scalfmm-mpi` and `scalfmm::scalfmm-starpu` (UNDER DEVELOPMENT) with distributed support.

## Building the Doc
The doc can be found [here](https://solverstack.gitlabpages.inria.fr/ScalFMM/) or you can build it locally.

You will need the following to build the doc :

    - Doxygen : install has you wish, through your package manager for example.
    - Sphinx : `pip install -U Sphinx`
    - Breathe : `pip install breathe`
    - Exhale : `pip install exhale`
    - Recommonmark : `pip install recommonmark`
    - Sphinx Read The Doc Theme : `pip install sphinx_rtd_theme`

```bash
cd path/to/build
cmake .. -Dscalfmm_BUILD_DOC=ON # or if cmake has already been called, ccmake .
make doxygen
make sphinx
```

This will generate the documentation in HTML format in the `build/doc/html` folder.

```bash
# From the Build folder
cd docs/sphinx
firefox index.html
```

A quick start file (~QUICKSTART.md~) is available, as well as user documentation. The latter is in orgmode, to generate the html you have to generate it by
```bash
emacs docs/user_guide.org  --batch -f org-html-export-to-html --kill
```
of for pdf file 
```bash
emacs docs/user_guide.org  --batch -f org-latex-export-to-pdf --kill
```

## Contributing and development guidelines

### Gitlab flow

Please, read the Gitlab flow article available [here](https://docs.gitlab.com/ee/workflow/gitlab_flow.html).

To make it simple, if you want to contribute to the library, create a branch from `experimental` with a meaningful name and develop
your feature in that branch. Keep your branch up to date by regularly rebasing your branch from the `experimental` branch to be up
to date. Once your are done, send a merge request.

### Branches

- `maintenance/scalfmm-1.5` and `maintenance/scalfmm-2.0` are main maintenance branches for older ScalFMM versions.
- `experimental` is the main branch for ScalFMM 3.0.



If you wish to contribute or test new feature, please branch from `experimental`.

## Folder structure
  - include: library core.
  - check: tools to check scalfmm functionalities.
  - data: particle distribution examples.
  - examples: common usage examples.
  - doc: documentation configuration.
  - units: unit tests.
  - tools: binaries to handle data files.
  - modules: dependencies required.
