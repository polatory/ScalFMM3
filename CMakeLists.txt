cmake_minimum_required (VERSION 3.18.0 FATAL_ERROR)
#
# Project Declaration
#--------------------
project(scalfmm CXX)

include(CMakePrintHelpers)

# check if compiling into source directories
string(COMPARE EQUAL "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" insource)
if(insource)
  message(FATAL_ERROR "${PROJECT_NAME} requires an out of source build. Goto ./Build and tapes cmake ../")
endif(insource)

#
# Version Number
# --------------
# ${CMAKE_PROJECT_NAME} version number.  An even minor number corresponds to releases.
set(${CMAKE_PROJECT_NAME}_MAJOR_VERSION 3)
set(${CMAKE_PROJECT_NAME}_MINOR_VERSION 0)
set(${CMAKE_PROJECT_NAME}_PATCH_VERSION 0-pre-alpha)
set(${CMAKE_PROJECT_NAME}_VERSION  "${${CMAKE_PROJECT_NAME}_MAJOR_VERSION}.${${CMAKE_PROJECT_NAME}_MINOR_VERSION}.${${CMAKE_PROJECT_NAME}_PATCH_VERSION}" )
set(CMAKE_CXX_STANDARD 17)

#
# Default build type
# ------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE
    STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Creating main lib
# -----------------
add_library(${CMAKE_PROJECT_NAME}-headers INTERFACE)
target_compile_features(${CMAKE_PROJECT_NAME}-headers INTERFACE cxx_std_17)

#
# Set library includes
# --------------------
target_include_directories(${CMAKE_PROJECT_NAME}-headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Set the RPATH config
# These options allow all installed executables to run correctly
# when installed to the install diretory
# --------------------
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#
# Flags for scalfmm
# -----------------
include(cmake/flags.cmake)

# -----------------
target_compile_options(${CMAKE_PROJECT_NAME}-headers INTERFACE "${${CMAKE_PROJECT_NAME}_CXX_FLAGS}")

# -----------------

#
# Options for scalfmm
# -------------------
include(cmake/options.cmake)

#
# Modules required
# ----------------
include(cmake/modules.cmake)

#
# Set ${CMAKE_PROJECT_NAME} compile definitions
# -------------------------------
set(${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS_HEADERS XSIMD_ENABLE_XTL_COMPLEX ENABLE_XTL_COMPLEX
  XTENSOR_USE_XSIMD CPP17 XTENSOR_FFTW_USE_FLOAT XTENSOR_FFTW_USE_DOUBLE HAVE_CBLAS=1 XTENSOR_DISABLE_EXCEPTIONS)

set(CORE_LIBRARIES BLAS LAPACK FFTW OpenMP)
set(${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS_HEADERS "${${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS_HEADERS}" CACHE STRING "Global compile definitions for ${CMAKE_PROJECT_NAME}-header")
message(STATUS "${CMAKE_PROJECT_NAME}-headers compile definitions : ${${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS_HEADERS}")

# -----------------
target_compile_definitions(${CMAKE_PROJECT_NAME}-headers INTERFACE "${${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS_HEADERS}")

# -----------------
#
# Dependencies for the lib
# ------------------------
set(FUSE_LIST "")
include(cmake/dependencies.cmake)

# Minimum required dependencies are BLAS/CBLAS and FFTW
target_link_libraries(${CMAKE_PROJECT_NAME}-headers INTERFACE cpp_tools::colors cpp_tools::timers)

set(${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS)
add_library(${CMAKE_PROJECT_NAME} INTERFACE)

if(${CMAKE_PROJECT_NAME}_BUILD_PBC)
  target_compile_definitions(${CMAKE_PROJECT_NAME} INTERFACE SCALFMM_BUILD_PBC)
  list(APPEND ${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS SCALFMM_BUILD_PBC)
endif()

# set(${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS "${${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS}"  STRING "Global compile definitions for ${CMAKE_PROJECT_NAME}")
message(STATUS "${CMAKE_PROJECT_NAME} compile definitions : ${${CMAKE_PROJECT_NAME}_COMPILE_DEFINITIONS}")

target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE ${CMAKE_PROJECT_NAME}-headers ${CBLAS_TARGET} ${FFTW_TARGET} ${OMP_TARGET})

if(${CMAKE_PROJECT_NAME}_USE_MPI)
  add_library(${CMAKE_PROJECT_NAME}-mpi INTERFACE)
  target_link_libraries(${CMAKE_PROJECT_NAME}-mpi INTERFACE ${CMAKE_PROJECT_NAME} ${MPI_TARGET})
  target_compile_definitions(${CMAKE_PROJECT_NAME}-mpi INTERFACE ${MPI_COMPILE_DEFINITIONS})
  set(MPI_LIBRARIES MPI)
endif()

message(STATUS "${CMAKE_PROJECT_NAME} dependencies : ${FUSE_LIST}")

#
# Set FUSE LIST
# -------------
set(FUSE_DEP_AVAILABLE "MPI;CUDA;STARPU;OMP;CBLAS;FFTW;OPENCL")

# List to compare against when compiling tests, etc...

#
# Modules required
# ----------------
include(cmake/analysis.cmake)

#
# Build Tools
# --------------
SET(SCALFMM_HEADERS_ONLY ON)
option(${CMAKE_PROJECT_NAME}_BUILD_TOOLS "Set to ON to build scalfmm3 tools." ON)
message(STATUS "${CMAKE_PROJECT_NAME}_BUILD_TOOLS = ${${CMAKE_PROJECT_NAME}_BUILD_TOOLS}")

if(${CMAKE_PROJECT_NAME}_BUILD_TOOLS)
  SET(SCALFMM_HEADERS_ONLY OFF)

  # Build - Examples and drivers
  add_subdirectory(tools)
endif()

#
# Build Examples
# --------------
option(${CMAKE_PROJECT_NAME}_BUILD_EXAMPLES "Set to ON to build scalfmm3 examples." ON)
message(STATUS "${CMAKE_PROJECT_NAME}_BUILD_EXAMPLES = ${${CMAKE_PROJECT_NAME}_BUILD_EXAMPLES}")

if(${CMAKE_PROJECT_NAME}_BUILD_EXAMPLES)
  set(SCALFMM_HEADERS_ONLY OFF)

  # Build - Examples and drivers
  add_subdirectory(examples)
endif()

#
# Build - Unitary and numerical Tests
# -----------------------------------
option(${CMAKE_PROJECT_NAME}_BUILD_UNITS "Set to ON to build scalfmm3 unit tests." OFF)
message(STATUS "${CMAKE_PROJECT_NAME}_BUILD_UNITS = ${${CMAKE_PROJECT_NAME}_BUILD_UNITS}")

if(${CMAKE_PROJECT_NAME}_BUILD_UNITS)
  enable_testing()
  include(CTest)
  add_subdirectory(units)
endif()

#
# Build - bench
# --------------
option(${CMAKE_PROJECT_NAME}_BUILD_BENCH "Set to ON to build scalfmm3 benchmarks." OFF)

if(${CMAKE_PROJECT_NAME}_BUILD_BENCH)
  set(SCALFMM_HEADERS_ONLY OFF)

  # Build - Bench
  add_subdirectory(bench)
endif()

# Build check/debug
# --------------
option(${CMAKE_PROJECT_NAME}_BUILD_CHECK "Set to ON to build scalfmm3 tools." ON)
message(STATUS "${CMAKE_PROJECT_NAME}_BUILD_TOOLS = ${${CMAKE_PROJECT_NAME}_BUILD_CHECK}")

if(${CMAKE_PROJECT_NAME}_BUILD_CHECK)
  # Build - Check
  add_subdirectory(checks)
endif()

#
# Add - doc
# ---------
option(${CMAKE_PROJECT_NAME}_BUILD_DOC "Set to ON to build the documentation" OFF)
message(STATUS "${CMAKE_PROJECT_NAME}_BUILD_DOC = ${${CMAKE_PROJECT_NAME}_BUILD_DOC}")

if(${CMAKE_PROJECT_NAME}_BUILD_DOC)
  add_subdirectory(docs)
endif()
#
# Export Library
# --------------
include(cmake/export.cmake)



