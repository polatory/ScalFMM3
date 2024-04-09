# Units
# -----

# List of source files
set(source_tests_files
    container/particle.cpp
    container/point_proxy.cpp

    container/variadic_containers.cpp
    tree/interaction_list.cpp
    operators/l2p.cpp

    # count kernel to check the number of interactions
    fmm/count_kernel.cpp
    fmm/count_kernel_st.cpp

    #
    fmm/non_homogenous.cpp
    fmm/homogenous.cpp
    fmm/non_mutual.cpp
    fmm/grad_one_over_r.cpp
    fmm/val_grad_one_over_r.cpp
    fmm/shift_grad_one_over_r.cpp
    fmm/logarithmic.cpp
    fmm/source-target.cpp

    # ompenmp check
    fmm/count_kernel_omp.cpp
    fmm/count_kernel_st_omp.cpp
    fmm/source-target_omp.cpp
    fmm/homogenous_omp.cpp
    fmm/shift_grad_one_over_r_omp.cpp
)

if(${CMAKE_PROJECT_NAME}_BUILD_PBC)
    list(APPEND source_tests_files
        fmm/periodic_level_0.cpp
        fmm/periodic_level_0_omp.cpp)
endif()

set(TEST_DATA_FILES_PATH ${CMAKE_SOURCE_DIR}/data/units/)
configure_file(fmm/units_fmm.hpp.in units_fmm.hpp @ONLY)
configure_file(fmm/units_source-target.hpp.in units_source-target.hpp @ONLY)

# Add execs - 1 cpp = 1 exec
foreach(exec ${source_tests_files})
    set(compile_exec TRUE)
    set(add_mpi_run FALSE)
    get_filename_component(execname ${exec} NAME_WE)

    foreach(fuse_key ${FUSE_DEP_AVAILABLE})
        file(STRINGS "${exec}" lines_fuse REGEX "@FUSE_${fuse_key}")

        if(lines_fuse AND NOT ${fuse_key} IN_LIST FUSE_LIST)
            message(STATUS "${exec} needs ${fuse_key}, dependencies are not met.")
            set(compile_exec FALSE)
        elseif(lines_fuse AND ${fuse_key} IN_LIST FUSE_LIST)
            list(APPEND TARGETS_TO_LINK ${${fuse_key}_TARGET})
            message(STATUS "${exec} unit activated with ${fuse_key} dependency.")

            if(${fuse_key}_COMPILE_DEFINITIONS)
                list(APPEND COMPILE_DEFINITIONS_TO_ADD ${${fuse_key}_COMPILE_DEFINITIONS})
            endif()
        endif()

        if(lines_fuse EQUAL "MPI")
            set(add_mpi_run TRUE)
        endif()
    endforeach()

    # Dependency are OK
    if(compile_exec)
        add_executable(unit.${execname} EXCLUDE_FROM_ALL ${exec})
        list(APPEND SCALFMM_UNITS_TARGETS unit.${execname})

        target_link_libraries(unit.${execname} ${CMAKE_PROJECT_NAME} ${TARGETS_TO_LINK})
        target_compile_definitions(unit.${execname} PRIVATE ${COMPILE_DEFINITIONS_TO_ADD})
        set(COMPILE_DEFINITIONS_TO_ADD "")
        target_include_directories(unit.${execname} PRIVATE ${CMAKE_SOURCE_DIR}/modules/external/catch2/ ${CMAKE_CURRENT_BINARY_DIR})
        set_target_properties(unit.${execname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_TYPE})

        if(add_mpi_run)
            add_test(NAME unit.${execname} COMMAND $<IF:${MPIEXEC_EXECUTABLE}, ${MPIEXEC_EXECUTABLE} -np 4 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/unit.${execname}>)
        else()
            add_test(NAME unit.${execname} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/unit.${execname})
        endif()
    endif()
endforeach(exec)

add_custom_target(units DEPENDS ${SCALFMM_UNITS_TARGETS})
set_target_properties(units PROPERTIES EXCLUDE_FROM_ALL True)
