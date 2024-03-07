#
# Export Library
# --------------
if(SCALFMM_HEADERS_ONLY)
set(TARGETS_TO_INSTALL ${CMAKE_PROJECT_NAME}-headers )
else()
set(TARGETS_TO_INSTALL ${CMAKE_PROJECT_NAME}-headers ${CMAKE_PROJECT_NAME})
endif()
if(TARGET ${CMAKE_PROJECT_NAME}-mpi)
    list(APPEND TARGETS_TO_INSTALL ${CMAKE_PROJECT_NAME}-mpi)
endif()

# MESSAGE(WARNING "TARGETS:  ${TARGETS_TO_INSTALL}")
install(TARGETS ${TARGETS_TO_INSTALL}
        EXPORT ${CMAKE_PROJECT_NAME}-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

install(DIRECTORY ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  )

include(cmake/export-cpptools)

install(DIRECTORY ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/modules/internal/inria_tools/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(DIRECTORY ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/modules/internal/xtl/include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(DIRECTORY ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/modules/internal/xsimd/include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(DIRECTORY ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/modules/internal/xtensor/include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(DIRECTORY ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/modules/internal/xtensor-blas/include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(DIRECTORY ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/modules/internal/xtensor-fftw/include/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(EXPORT ${CMAKE_PROJECT_NAME}-targets
        FILE ${CMAKE_PROJECT_NAME}-targets.cmake
        NAMESPACE ${CMAKE_PROJECT_NAME}::
        DESTINATION lib/cmake/${CMAKE_PROJECT_NAME})


include(CMakePackageConfigHelpers)

write_basic_package_version_file("${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${${CMAKE_PROJECT_NAME}_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_PROJECT_NAME}Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/${CMAKE_PROJECT_NAME}/)

install(FILES "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
  DESTINATION lib/cmake/${CMAKE_PROJECT_NAME})

foreach(file ${TOOLS_TO_INSTALL})
    install(PROGRAMS "${CMAKE_BINARY_DIR}/tools/${CMAKE_BUILD_TYPE}/${file}"
    DESTINATION bin/)
endforeach() ## build a CPack driven installer package
# --------------------------------------
# See CPackConfig.cmake for the details.
include(CPack)
