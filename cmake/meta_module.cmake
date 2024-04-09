#
# meta-module cmake
# -----------------

function(meta_module MODULE_NAME MODULE_GIT_URL MODULE_GIT_TAG MODULE_CMAKE_ARGS)
    message(STATUS "Module ${MODULE_NAME} git download url : ${MODULE_GIT_URL}")
    message(STATUS "Module ${MODULE_NAME} git tag : ${MODULE_GIT_TAG}")
    message(STATUS "Module ${MODULE_NAME} cmake arguments : ${MODULE_CMAKE_ARGS}")
    configure_file(cmake/meta_module_template.cmake.in ${CMAKE_BINARY_DIR}/modules/${MODULE_NAME}/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/modules/${MODULE_NAME}/)
    if(result)
        message(FATAL_ERROR "Configure step for ${MODULE_NAME} failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_MAKE_PROGRAM} ${MODULE_NAME}
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/modules/${MODULE_NAME}/)
    if(result)
        message(FATAL_ERROR "Build step for ${MODULE_NAME} failed: ${result}")
    endif()
endfunction()

function(deploy_module MODULE_NAME MODULE_GIT_URL MODULE_GIT_TAG MODULE_CMAKE_ARGS MODULE_INSTALL_PATH MODULE_TARGET)
    find_package(${MODULE_NAME} QUIET)
    if(${MODULE_NAME}_FOUND)
        message(STATUS "Local ${MODULE_NAME} found")
        target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE ${MODULE_NAME})
    else(${MODULE_NAME}_FOUND)
        message(STATUS "${MODULE_NAME} is required : trigering meta-package.")

        meta_module(${MODULE_NAME} ${MODULE_GIT_URL} ${MODULE_GIT_TAG} ${MODULE_CMAKE_ARGS})

        set(ENV{${MODULE_TARGET}_DIR} ${MODULE_INSTALL_PATH})
        find_package(${MODULE_TARGET} REQUIRED PATHS ${MODULE_INSTALL_PATH})

        if(${MODULE_TARGET}_FOUND)
            message(STATUS "Meta package ${MODULE_NAME} found")
            target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE ${MODULE_TARGET})
            if(TARGET ${MODULE_TARGET}::${MODULE_TARGET})
                message(${MODULE_TARGET})
                install(TARGETS ${MODULE_TARGET}
                    PUBLIC_HEADER DESTINATION $<INSTALL_INTERFACE:include>
                    LIBRARY DESTINATION $<INSTALL_INTERFACE:lib>
                    )
            endif()
        else(${MODULE_TARGET}_FOUND)
            message(FATAL_ERROR "Meta package ${MODULE_NAME} can't be found. Automatic install failed.")
        endif(${MODULE_TARGET}_FOUND)
    endif(${MODULE_NAME}_FOUND)
endfunction()
