#
# dependencies for scalfmm
# ------------------------

include(cmake/dependencies/openmp.cmake)
include(cmake/dependencies/mpi.cmake)
include(cmake/dependencies/blas_lapack.cmake)
include(cmake/dependencies/fftw.cmake)
include(cmake/dependencies/starpu.cmake)

if(${CMAKE_PROJECT_NAME}_USE_CATALYST)
    find_package(catalyst REQUIRED)
    if(catalyst_FOUND)
        message(STATUS "Catalyst found.")
        target_link_libraries(${CMAKE_PROJECT_NAME}-headers INTERFACE catalyst::catalyst catalyst::conduit_headers catalyst::conduit catalyst::blueprint)
    endif()
endif()

