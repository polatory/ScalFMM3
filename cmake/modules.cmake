#
# XTL, XSIMD & XTENSOR
# --------------------
# Find XSIMD properly

#
# Module internal
# ------------
target_include_directories(${CMAKE_PROJECT_NAME}-headers SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/internal/inria_tools>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${CMAKE_PROJECT_NAME}-headers SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/internal/xtl/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${CMAKE_PROJECT_NAME}-headers SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/internal/xsimd/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${CMAKE_PROJECT_NAME}-headers SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/internal/xtensor/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${CMAKE_PROJECT_NAME}-headers SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/internal/xtensor-blas/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(${CMAKE_PROJECT_NAME}-headers SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/internal/xtensor-fftw/include>
    $<INSTALL_INTERFACE:include>
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules/internal/cpp_tools/)
set(CPP_TOOLS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/modules/internal/cpp_tools/)

set(CPP_TOOLS_USE_CL_PARSER ON)
set(CPP_TOOLS_USE_COLORS ON)
set(CPP_TOOLS_USE_TIMERS ON)

set(CPP_TOOLS_USE_PARALLEL_MANAGER ON)

include(cmake/init-cpptools)
