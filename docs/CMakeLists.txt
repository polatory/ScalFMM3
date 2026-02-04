#
# Docs
# ----

find_package(Doxygen REQUIRED)

# Look for an executable called sphinx-build
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Path to sphinx-build executable")

include(FindPackageHandleStandardArgs)

# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Sphinx
    "Failed to find sphinx-build executable"
    SPHINX_EXECUTABLE)
file(GLOB_RECURSE SCALFMM_HEADERS
    ${CMAKE_SOURCE_DIR}/include/*.hpp
    ${CMAKE_SOURCE_DIR}/modules/internal/inria_tools/inria/*.hpp
)

set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include)
set(DOXYGEN_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)
set(DOXYGEN_MODULES_DIR ${PROJECT_SOURCE_DIR}/modules/internal/inria_tools/inria)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) # Doxygen won't create this for us

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${SCALFMM_HEADERS}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
    COMMENT " Generating docs ")

add_custom_target(doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/docs/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
set(RST_FILES
    index.rst
)

add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
    COMMAND
    ${SPHINX_EXECUTABLE} -b html

    # Tell Breathe where to find the Doxygen output
    -Dbreathe_projects.scalfmm3=${DOXYGEN_OUTPUT_DIR}/xml
    ${SPHINX_SOURCE} ${SPHINX_BUILD}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS

    # Other docs files you want to track should go here (or in some variable)
    ${RST_FILES}
    ${DOXYGEN_INDEX_FILE}

    # ${CMAKE_CURRENT_SOURCE_DIR}
    MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
    COMMENT " Generating documentation with Sphinx ")

# Nice named target so we can run the job easily
add_custom_target(sphinx  DEPENDS ${SPHINX_INDEX_FILE}) # add_${TARGET_DEPENDS})# Add an install target to install the docs
install(DIRECTORY ${SPHINX_BUILD}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/docs)
    
   # Nice named target so we can run the job easily
add_custom_target(doc ALL DEPENDS sphinx)