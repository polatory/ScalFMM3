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
  - Custom BLAS implementations (MKL, OpenBlas, ...)
  - [FFTW3](https://www.fftw.org/) library.

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
### The different branches in ScalFMM
 - main latest development
 - master-2.0 the master of the 2.0 version (no evolution)
 - maintenance (the official version )
   - scalfmm-3.0 the release of the 3.0 version
   - scalfmm-2.0 the release of the 2.0 version (no evolution)
   - scalfmm-1.5 the release of the 1.5 version (no evolution)

### Features
   - scalfmm-3.0 the release of the 3.0 version
     - barycentric,  Chebyshev, uniform interpolation
     - uniform tree
     - task based OpenMP,MPI
   - scalfmm-2.0 the release of the 2.0 version
     - adaptive and uniform tree
     - Chebyshev, uniform interpolation
     - MPI, OpenMP, StarPU (Chebyshev P2P and M2L operators for GPU)
   - scalfmm-1.5 the release of the 1.5 version
     - uniform tree
     - Chebyshev, uniform interpolation
     - MPI, OpenMP, StarPU (Chebyshev P2P and M2L operators for GPU)
### Cloning

To use the last development states of ScalFMM, please clone the main
branch. Note that ScalFMM contains multiple git submodules like `morse_cmake` and the `xtensor` software stack.
  To get sources please use these commands:
```bash
git clone --recursive https://gitlab.inria.fr/solverstack/ScalFMM.git
```
or
```bash
git clone https://gitlab.inria.fr/solverstack/ScalFMM.git
cd ScalFMM
git submodule init
git submodule update

```
### Building
The project has been tested  with the following compiler  both on Linux and OSX
  - GCC 11 and above
  - LLVM 17 and above

ScalFMM uses iterators on task dependencies (OpenMP 5.x) the following compilers do not support them yet
  - Intel OneAPI 2022 and lower

To keep a clean source tree, do an out-of-source build by creating a `build` folder out of your clone.
#### OpenMP or sequential installation

##### On Linux
```bash
cd /path/to/build/
# Use cmake, with relevant options
cmake .. # -Dscalfmm_USE_MKL=ON
```

The build may be configured after the first CMake invocation using, for instance, `ccmake` or `cmake-gui`.

```bash
# Still in the build folder
ccmake .
# or
cmake-gui .
```
##### On macosX

To use Clang's native compiler, you must first install the omp library and use the `-Xclang -fopenmp` compilation flags.
```bash
brew install libomp
```
and to compile
```bash
cd /path/to/build/
# Use cmake, with relevant options
cmake  -DCMAKE_CXX_FLAGS=`-Xclang -fopenmp`  -S ../ 
```
#### Optimization

Customization or optimized options:
  - If you plan to use Intel MKL, please set `scalfmm_USE_MKL` to `ON`. This will also choose MKL for the FFTs and prevent unusual behavior with multithreaded MKL version.
  - To use a specific version of labpack/blas you can specify it using the BLA_VENDOR variable in cmake. For example  `-DBLA_VENDOR=OpenBLAS` for OpenBLAS lapack and blas library
  - if we are interested in periodic boundary condition, the option `scalfmm_BUILD_PBC` has to be set to `ON`.
  -
The binaries are then compiled by calling `make` (or `ninja` if you specified it at the configure step).

The `all` target only build the tools binaries. These are binaries that can generate particle distributions,
perform direct fmm computation, etc.

Invoke `make help` to see the available targets.
Global targets are available :
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
If you are targeting a specific architecture, add the appropriate SIMD flag to your version flags.
For example, if you are targeting an AVX2 processor, add `-mavx2` to your flags.
And if you know the microarchitecture, such as cascade lake for example, you can also add the `-march=cascadelake` flag.
When you add architecture flags, SIMD flags are automatically triggered.

## Using ScalFMM in your project
You can integrate ScalFMM as a package after its intalation or as an header only library.
###  with find_package 
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

###  header only 
It's a bit trickier, remembering that ScalFMM depends on OpenMP, Lapack/Blas and fftw. 
The first step is to define a scalfmm target as follows 
```cmake
set(SCALFMM_DIR "$ENV{SCALFMM_DIR}")
add_library(scalfmm INTERFACE)
target_compile_definitions(scalfmm INTERFACE ENABLE_XTL_COMPLEX HAVE_CBLAS=1 XSIMD_ENABLE_XTL_COMPLEX XTENSOR_DISABLE_EXCEPTIONS XTENSOR_FFTW_USE_DOUBLE XTENSOR_FFTW_USE_FLOAT XTENSOR_USE_XSIMD)
target_compile_features(scalfmm INTERFACE cxx_std_17)
target_include_directories(scalfmm INTERFACE "${SCALFMM_DIR}/include" "${SCALFMM_DIR}/modules/internal/inria_tools"  "${SCALFMM_DIR}/modules/internal/xtensor-fftw/include"
"${SCALFMM_DIR}/modules/internal/cpp_tools/colors/include" "${SCALFMM_DIR}/modules/internal/cpp_tools/timers/include"
"${SCALFMM_DIR}/modules/internal/cpp_tools/cl_parser/include" "${SCALFMM_DIR}/modules/internal/xsimd/include"
"${SCALFMM_DIR}/modules/internal/xtensor/include" "${SCALFMM_DIR}/modules/internal/xtensor-blas/include"
"${SCALFMM_DIR}/modules/internal/xtl/include"
)
``` 
Then 
```cmake
include (FindPkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING " OPENMP NOT FOUND ")
    target_compile_definitions(scalfmm INTERFACE SCAL_USE_SEQ)
endif(NOT OpenMP_CXX_FOUND)

pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
pkg_search_module(FFTWF REQUIRED fftw3f IMPORTED_TARGET)
list(APPEND FFTW_TARGET PkgConfig::FFTW)
list(APPEND FFTW_TARGET PkgConfig::FFTWF)

set(BLA_VENDOR "OpenBLAS")
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
```

Finally, to build `your_target` simply add the scalfmm target to `target_link_libraries` as follows
```cmake
target_link_libraries(your-target scalfmm LAPACK::LAPACK BLAS::BLAS   ${FFTW_TARGET} OpenMP::OpenMP_CXX )
```


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
make doc
```

This will generate the documentation in HTML format in the `build/docs/sphinx` folder.

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

## Using Guix

Work in progress...

### Workflow for developers

#### Using OpenBLAS

``` bash
guix shell -C -D scalfmm --preserve="^TERM$"
cmake -B build -DBLA_VENDOR=OpenBLAS -Dscalfmm_BUILD_UNITS=ON
cmake --build build --target units
ctest --test-dir build       # run the unit tests
```

#### Using Intel MKL

``` bash
guix shell -C -D scalfmm-mkl --preserve="^TERM$"
export MKLROOT=$GUIX_ENVIRONMENT  # to help cmake to find the required *.cmake files
cmake -B build -Dscalfmm_USE_MKL=ON -Dscalfmm_BUILD_UNITS=ON
cmake --build build --target units
ctest --test-dir build       # run the unit tests
```

### Workflow for users

``` bash
guix shell -C scalfmm <other packages>
```

Alternatively (using Intel MKL)

``` bash
guix shell -C scalfmm-mkl <other packages>
```

### Installation via guix

``` bash
guix install scalfmm
```

Alternatively (using Intel MKL)

``` bash
guix install scalfmm-mkl
```

### Run reproducible benchmarks

TODO: provide instructions to run reproducible benchmarks using `guix time-machine` and `guix shell`

``` bash
guix time-machine -C .guix/scalfmm-channels.scm -- shell -C -m .guix/scalfmm-manifest-openblas.scm
```

We provide several manifest files: `scalfmm-manifest-openblas.scm`, `scalfmm-manifest-mkl.scm`.

### Build the documentation with guix

TODO: provide instructions to build the documentations using `guix time-machine` and `guix shell`

``` bash
guix time-machine -C .guix/scalfmm-channels.scm -- shell -C -m .guix/scalfmm-manifest-doc.scm -L .guix/
```

## Contributing and development guidelines

### Gitlab flow

Please, read the Gitlab flow article available [here](https://docs.gitlab.com/ee/workflow/gitlab_flow.html).

To make it simple, if you want to contribute to the library, create a branch from `experimental` with a meaningful name and develop
your feature in that branch. Keep your branch up to date by regularly rebasing your branch from the `experimental` branch to be up
to date. Once you are done, send a merge request.

### Branches

- `maintenance/scalfmm-1.5` and `maintenance/scalfmm-2.0` are main maintenance branches for older ScalFMM versions.
- `experimental` is the main branch for ScalFMM 3.0.



If you wish to contribute or test new features, please branch from `experimental`.

## Folder structure
  - include: library core.
  - check: tools to check scalfmm functionalities.
  - data: particle distribution examples.
  - examples: common usage examples.
  - doc: documentation configuration.
  - units: unit tests.
  - tools: binaries to handle data files.
  - modules: dependencies required.
