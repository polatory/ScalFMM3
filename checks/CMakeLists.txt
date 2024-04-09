# check
# ---------

# List of source files
set(source_check_files
    check_morton_sort.cpp
    check_morton_index.cpp
    check_interaction_lists.cpp

    # # debug & check
    count_particles_seq.cpp
    count_particles_st_seq.cpp
    count_particles_omp.cpp
    count_particles_st_omp.cpp

    #
    check_interpolation.cpp
    check_l2p.cpp
    check_p2p.cpp

    #
    test_chebyshev.cpp
    test_m2l_sym.cpp
    debug_omp.cpp
    test_block.cpp
    test_build_tree.cpp

    #
    check_1d.cpp
    check_2d.cpp

    # xtensor blas
    check_xtensor_blas.cpp

    # Diwina
    check_source-target.cpp

    test_p2p_inner_non_mutual.cpp
)

if(${CMAKE_PROJECT_NAME}_BUILD_PBC)
    list(APPEND source_check_files
        check_periodic.cpp
        check_periodic_omp.cpp)
endif()

if(${CMAKE_PROJECT_NAME}_USE_MPI)
    list(APPEND source_check_files
        test_build_let.cpp
        check_mpi.cpp

        # test_compose.cpp
    )
    # message(WARNING "source_check_files ")
    # cmake_print_variables(source_check_files)
endif()

# Add execs - 1 cpp = 1 exec
foreach(exec ${source_check_files})
    set(compile_exec TRUE)
    get_filename_component(execname ${exec} NAME_WE)

    foreach(fuse_key ${FUSE_DEP_AVAILABLE})
        file(STRINGS "${exec}" lines_fuse REGEX "@FUSE_${fuse_key}")
        if(lines_fuse AND NOT ${fuse_key} IN_LIST FUSE_LIST)
            message(STATUS "${exec} needs ${fuse_key}, dependencies are not met.")
            set(compile_exec FALSE)
        elseif(lines_fuse AND ${fuse_key} IN_LIST FUSE_LIST)
            message(STATUS "${exec} tool activated with ${fuse_key}.")
            list(APPEND TARGETS_TO_LINK ${${fuse_key}_TARGET})

            if(${fuse_key}_COMPILE_DEFINITIONS)
                list(APPEND COMPILE_DEFINITIONS_TO_ADD ${${fuse_key}_COMPILE_DEFINITIONS})
            endif()
        endif()
    endforeach()

    # Dependency are OK
    if(compile_exec)
        list(APPEND ${CMAKE_PROJECT_NAME}_CHECK_TARGETS ${execname})
        add_executable(${execname} EXCLUDE_FROM_ALL ${exec})
        target_link_libraries(${execname} ${CMAKE_PROJECT_NAME} cpp_tools::cl_parser ${TARGETS_TO_LINK})
#        target_include_directories(${execname} PRIVATE ${PROJECT_SOURCE_DIR}/examples)
        target_compile_definitions(${execname} PRIVATE ${COMPILE_DEFINITIONS_TO_ADD})
        set(COMPILE_DEFINITIONS_TO_ADD "")
        set_target_properties(${execname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_TYPE})
    endif()
endforeach(exec)

add_custom_target(checks DEPENDS ${${CMAKE_PROJECT_NAME}_CHECK_TARGETS})
set_target_properties(checks PROPERTIES EXCLUDE_FROM_ALL True)
