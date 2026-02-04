#
# Binding (pybind11)
# -------------------


# List of source files
set(source_files
    pyfmm.cpp
)
set(CMAKE_CXX_STANDARD 20)

foreach(exec ${source_files})
    get_filename_component(execname ${exec} NAME_WE)

    list(APPEND ${CMAKE_PROJECT_NAME}_PYTHON_INTERFACE_TARGETS ${execname})
    pybind11_add_module(${execname} EXCLUDE_FROM_ALL ${exec})
    target_link_libraries(${execname} PRIVATE scalfmm)
    target_include_directories(${execname} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${execname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_TYPE})

    # Link-Time Optimization (LTO)
    # The following option allows the compiler to determine the number of jobs to run during the Link-Time Optimization (LTO) phase automatically based on the available hardware resources
    # if(CMAKE_BUILD_TYPE MATCHES Release)
    #     target_compile_options(${execname} PRIVATE -flto=auto)
    # endif()
endforeach(exec)

add_custom_target(python_interface DEPENDS ${${CMAKE_PROJECT_NAME}_PYTHON_INTERFACE_TARGETS})
set_target_properties(python_interface PROPERTIES EXCLUDE_FROM_ALL True)