# Examples
# ---------
# List of source files
set(source_tests_files

    # test
    test_particles.cpp
    test_dimension.cpp
    test_dimension_low_rank.cpp
    test_dimension_omp.cpp

    # FMM
    test_laplace_kernels.cpp
    test_like_mrhs.cpp

    # # debug & check
    # count_particles_seq.cpp
    # count_particles_st_seq.cpp
    # count_particles_omp.cpp
    # count_particles_st_omp.cpp

    # test_time_loop.cpp
    # test to move in compose/sandox project
    fmm_source_target.cpp
    tutorial.cpp
)

if(${CMAKE_PROJECT_NAME}_USE_MPI)
    list(APPEND source_tests_files
        test_mpi_algo.cpp
        count_particles_mpi.cpp
    )
endif()

if(${CMAKE_PROJECT_NAME}_BUILD_PBC)
    list(APPEND source_tests_files
        test_periodic_dist.cpp)
endif()
# Add execs - 1 cpp = 1 exec
foreach(exec ${source_tests_files})
    set(compile_exec TRUE)
    get_filename_component( execname ${exec} NAME_WE )

    file(STRINGS "${exec}" fuse_mpi REGEX "@FUSE_MPI")
    if(fuse_mpi AND TARGET ${CMAKE_PROJECT_NAME}-mpi)
        set(TARGET_TO_LINK scalfmm-mpi)
        message( STATUS "${exec} example activated with MPI dependency." )
    elseif(fuse_mpi AND NOT TARGET ${CMAKE_PROJECT_NAME}-mpi)
        message( STATUS "${exec} example not activated with MPI dependency." )
        set(compile_exec FALSE)
    else()
        set(TARGET_TO_LINK scalfmm)
    endif()

    # Dependency are OK
    if( compile_exec )
        list(APPEND ${CMAKE_PROJECT_NAME}_EXAMPLES_TARGETS ${execname})
        add_executable( ${execname} EXCLUDE_FROM_ALL ${exec})
        target_link_libraries( ${execname} ${TARGET_TO_LINK} cpp_tools::cl_parser)
        target_compile_definitions( ${execname} PRIVATE ${COMPILE_DEFINITIONS_TO_ADD})
        set(COMPILE_DEFINITIONS_TO_ADD "")
        set_target_properties(${execname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_TYPE})
    endif()
    unset(TARGET_TO_LINK)
endforeach(exec)

add_custom_target(examples DEPENDS ${${CMAKE_PROJECT_NAME}_EXAMPLES_TARGETS})
set_target_properties(examples PROPERTIES EXCLUDE_FROM_ALL True)

