# Tools
# ---------

# List of source files
set(source_tests_files

    # tools
    change_file_format.cpp
    sort_particles.cpp
    compare_files.cpp
    generate_distribution.cpp
    direct_computation.cpp

    # define_symmetries.cpp
    trace_tree.cpp
    compare_trees.cpp
)

# Add execs - 1 cpp = 1 exec
foreach(exec ${source_tests_files})
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
        add_executable(${execname} EXCLUDE_FROM_ALL ${exec})
        list(APPEND ${CMAKE_PROJECT_NAME}_TOOLS_TARGETS ${execname})
        target_link_libraries(${execname} ${CMAKE_PROJECT_NAME} cpp_tools::cl_parser ${TARGETS_TO_LINK})
        target_compile_definitions(${execname} PRIVATE ${COMPILE_DEFINITIONS_TO_ADD})
        set_target_properties(${execname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_TYPE})
    endif()
endforeach(exec)

add_custom_target(tools ALL DEPENDS ${${CMAKE_PROJECT_NAME}_TOOLS_TARGETS})
set(TOOLS_TO_INSTALL ${${CMAKE_PROJECT_NAME}_TOOLS_TARGETS} CACHE INTERNAL "" FORCE)
