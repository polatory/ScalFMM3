#
# STARPU Section
# --------------
if( ${CMAKE_PROJECT_NAME}_USE_STARPU )
    # TODO
    #enable_language(C)
    ## Find StarPU with a list of optional components
    #set(SCALFMM_STARPU_VERSION "1.2.8" CACHE STRING " STARPU version desired")
    ## create list of components in order to make a single call to find_package(starpu...)
    #set(STARPU_COMPONENT_LIST "HWLOC")
    #if(SCALFMM_USE_CUDA)
    #  message(WARNING "This release doest not include a full support for CUDA/OpenCl.")
    #  #list(APPEND STARPU_COMPONENT_LIST "CUDA")
    #endif()
    #if(SCALFMM_USE_MPI)
    #  list(APPEND STARPU_COMPONENT_LIST "MPI")
    #endif()
    #if(SCALFMM_USE_OPENCL)
    #  message(WARNING "This release doest not include a full support for CUDA/OpenCl.")
    #  #list(APPEND STARPU_COMPONENT_LIST "OPENCL")
    #endif()

    #find_package(STARPU ${SCALFMM_STARPU_VERSION} REQUIRED
    #  COMPONENTS ${STARPU_COMPONENT_LIST})

    #if(STARPU_FOUND)
    #  target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE starpu::starpu_dep)
    #  list(APPEND FUSE_LIST "STARPU")
    #  list(APPEND SCALFMM_LIBRARIES "STARPU")
    #  if(SCALFMM_USE_CUDA)
    #    #list(APPEND FUSE_LIST "CUDA")
    #  endif()
    #  if(SCALFMM_USE_OPENCL)
    #    #list(APPEND FUSE_LIST "OPENCL")
    #  endif()
    #else(STARPU_FOUND)
    #  message(FATAL_ERROR "StarPU not found.")
    #endif(STARPU_FOUND)
endif(${CMAKE_PROJECT_NAME}_USE_STARPU)

