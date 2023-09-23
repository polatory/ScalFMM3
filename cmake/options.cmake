#
# Options for scalfmm
# -------------------

# Cmake options for features 
option( ${CMAKE_PROJECT_NAME}_BUILD_PBC              "Set to ON to build ScalFMM with preriodic Boundary condition"        OFF )

# Cmake options for dependencies
option( ${CMAKE_PROJECT_NAME}_USE_MPI              "Set to ON to build ScalFMM with MPI"         OFF )
option( ${CMAKE_PROJECT_NAME}_USE_STARPU           "Set to ON to build ${CMAKE_PROJECT_NAME} with StarPU"     OFF )
option( ${CMAKE_PROJECT_NAME}_USE_MKL              "Set to ON to build ScalFMM with MKL"        OFF )
#option( ${CMAKE_PROJECT_NAME}_USE_ESSL              "Set to ON to build ScalFMM with ESSL"        OFF )

# Cmake options related to trace, logging and statistics
option( ${CMAKE_PROJECT_NAME}_USE_LOG              "Set to ON to print output debug information"  OFF )
option( ${CMAKE_PROJECT_NAME}_USE_MEM_STATS        "Set to ON to profile memory"                  OFF )
option( ${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS  "Set to ON to use static analysis" OFF )



include(CMakeDependentOption)

cmake_dependent_option(${CMAKE_PROJECT_NAME}_USE_OPENCL "Set to ON to use OPENCL with StarPU" OFF "${CMAKE_PROJECT_NAME}_USE_STARPU" OFF)
cmake_dependent_option(${CMAKE_PROJECT_NAME}_USE_CUDA "Set to ON to use OPENCL with StarPU" OFF "${CMAKE_PROJECT_NAME}_USE_STARPU" OFF)

cmake_dependent_option(${CMAKE_PROJECT_NAME}_USE_CPPCHECK   "Set to ON to use static analysis tools"  OFF "${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS"  OFF )
cmake_dependent_option(${CMAKE_PROJECT_NAME}_USE_CLANGTIDY  "Set to ON to use static analysis tools"  ON  "${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS"  OFF  )
cmake_dependent_option(${CMAKE_PROJECT_NAME}_USE_IWYU       "Set to ON to use static analysis tools"  ON  "${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS"  OFF )

# Additional options for developers


message( STATUS "${CMAKE_PROJECT_NAME}_BUILD_PBC     =  ${${CMAKE_PROJECT_NAME}_BUILD_PBC}")

message( STATUS "${CMAKE_PROJECT_NAME}_USE_MPI    =  ${${CMAKE_PROJECT_NAME}_USE_MPI}")
message( STATUS "${CMAKE_PROJECT_NAME}_USE_STARPU =  ${${CMAKE_PROJECT_NAME}_USE_STARPU}")
message( STATUS "${CMAKE_PROJECT_NAME}_USE_MKL    =  ${${CMAKE_PROJECT_NAME}_USE_MKL}")
#message( STATUS "${CMAKE_PROJECT_NAME}_USE_ESSL    =  ${${CMAKE_PROJECT_NAME}_USE_ESSL}")

# Cmake options related to trace, logging and statistics
message( STATUS "${CMAKE_PROJECT_NAME}_USE_LOG       = ${${CMAKE_PROJECT_NAME}_USE_LOG}")
message( STATUS "${CMAKE_PROJECT_NAME}_USE_MEM_STATS = ${${CMAKE_PROJECT_NAME}_USE_MEM_STATS}")
message( STATUS "${CMAKE_PROJECT_NAME}_ONLY_DEVEL    = ${${CMAKE_PROJECT_NAME}_ONLY_DEVEL}")

if(${CMAKE_PROJECT_NAME}_USE_STARPU)
    message( STATUS "${CMAKE_PROJECT_NAME}_USE_CUDA   = ${${CMAKE_PROJECT_NAME}_USE_CUDA}")
    message( STATUS "${CMAKE_PROJECT_NAME}_USE_OPENCL = ${${CMAKE_PROJECT_NAME}_USE_OPENCL}")
endif(${CMAKE_PROJECT_NAME}_USE_STARPU)

message( STATUS "${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS = ${${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS}" )
message( STATUS "${CMAKE_PROJECT_NAME}_USE_CPPCHECK        = ${${CMAKE_PROJECT_NAME}_USE_CPPCHECK}" )
message( STATUS "${CMAKE_PROJECT_NAME}_USE_CLANGTIDY       = ${${CMAKE_PROJECT_NAME}_USE_CLANGTIDY}" )
message( STATUS "${CMAKE_PROJECT_NAME}_USE_IWYU            = ${${CMAKE_PROJECT_NAME}_USE_IWYU}" )
