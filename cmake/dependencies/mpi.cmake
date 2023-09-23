#
# MPI
# ---
if(${CMAKE_PROJECT_NAME}_USE_MPI)
    if(NOT MPI_CXX_FOUND)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        list(APPEND MPI_TARGET MPI::MPI_CXX cpp_tools::parallel_manager)
            list(APPEND MPI_COMPILE_DEFINITIONS CPP_TOOLS_PARALLEL_MANAGER_USE_MPI)
            list(APPEND MPI_COMPILE_DEFINITIONS SCALFMM_USE_MPI)
        list(APPEND FUSE_LIST MPI)
    else(MPI_CXX_FOUND)
      message(FATAL_ERROR "MPI_CXX is required but was not found. "
              "Please provide an MPI compiler in your environment."
              "Note that you can give the path to your MPI installation "
              "by setting MPI_DIR cmake variable.")
    endif(MPI_CXX_FOUND)
endif(NOT MPI_CXX_FOUND)
endif(${CMAKE_PROJECT_NAME}_USE_MPI)
