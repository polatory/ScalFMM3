# Benchmarks
# ---------
# List of source files
set(source_tests_files
    tasks_dep_omp.cpp
    sequential.cpp
    loop_leaf.cpp
)

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
        list(APPEND ${CMAKE_PROJECT_NAME}_BENCH_TARGETS ${execname})
        add_executable( ${execname} EXCLUDE_FROM_ALL ${exec})
        target_link_libraries( ${execname} ${TARGET_TO_LINK} cpp_tools::cl_parser)
        target_compile_definitions( ${execname} PRIVATE ${COMPILE_DEFINITIONS_TO_ADD})
        set(COMPILE_DEFINITIONS_TO_ADD "")
        set_target_properties(${execname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_TYPE})
    endif()
    unset(TARGET_TO_LINK)
endforeach(exec)

add_custom_target(bench DEPENDS ${${CMAKE_PROJECT_NAME}_BENCH_TARGETS})
set_target_properties(bench PROPERTIES EXCLUDE_FROM_ALL True)

