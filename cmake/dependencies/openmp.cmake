#
# OpenMP
# ------

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    list(APPEND OMP_TARGET OpenMP::OpenMP_CXX cpp_tools::parallel_manager)
    list(APPEND OMP_COMPILE_DEFINITIONS CPP_TOOLS_PARALLEL_MANAGER_USE_OMP)
    list(APPEND FUSE_LIST OMP)
    # cmake_print_variables(CMAKE_CXX_COMPILER_ID)
    # cmake_print_variables(CMAKE_CXX_COMPILER_VERSION)
    # cmake_print_variables(CMAKE_CXX_COMPILER_VERSION_INTERNAL)
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "16.0")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
# endif()
# endif()
else(OpenMP_CXX_FOUND)
    message(WARNING "OPENMP NOT FOUND")
endif(OpenMP_CXX_FOUND)
