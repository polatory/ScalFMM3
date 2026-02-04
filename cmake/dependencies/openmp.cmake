#
# OpenMP
# ------
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    # https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
    # build the target
    find_file(OMP_H omp.h ENV HOMEBREW_CELLAR PATH_SUFFIXES "libomp/18.1.6/include")
    cmake_print_variables(OMP_H)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_COMPILE_OPTIONS "-Xclang -fopenm")
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/usr/local/Cellar/libomp/18.1.6/include")
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_COMPILE_DEFINITIONS "_OPENMP")
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_LINK_DIRECTORIES "/usr/local/Cellar/libomp/18.1.6/lib")

    # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
    set_property(TARGET OpenMP::OpenMP_CXX
        PROPERTY INTERFACE_LINK_LIBRARIES -lomp)
    set(OpenMP_CXX_FOUND ON)
else()
    find_package(OpenMP REQUIRED)
endif()

if(OpenMP_CXX_FOUND)
    list(APPEND OMP_TARGET OpenMP::OpenMP_CXX cpp_tools::parallel_manager)
    list(APPEND OMP_COMPILE_DEFINITIONS CPP_TOOLS_PARALLEL_MANAGER_USE_OMP)
    list(APPEND FUSE_LIST OMP)

else(OpenMP_CXX_FOUND)
    message(WARNING " OPENMP NOT FOUND ")
endif(OpenMP_CXX_FOUND)

cmake_print_variables(FUSE_LIST)