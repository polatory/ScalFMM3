#
# Flags for scalfmm
# -----------------

list(APPEND ${CMAKE_PROJECT_NAME}_CXX_FLAGS -funroll-loops)
# Warning : -ffast-math was used, danger is in trigonometric functions
# where agressive floating point optimisations lead to unsafe math operations.
list(APPEND ${CMAKE_PROJECT_NAME}_CXX_FLAGS -ftree-vectorize)

#
# dev flags
# ---------
if(${CMAKE_PROJECT_NAME}_ONLY_DEVEL)
    list(APPEND ${CMAKE_PROJECT_NAME}_CXX_FLAGS -Wall -pedantic )
endif(${CMAKE_PROJECT_NAME}_ONLY_DEVEL)

#
# Specific Debug flags
# --------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    list(APPEND ${CMAKE_PROJECT_NAME}_CXX_FLAGS -fp-model\ strict)
endif()

#
# Specific Release flags
# ----------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    list(APPEND ${CMAKE_PROJECT_NAME}_CXX_FLAGS  -ipo  -fstrict-aliasing)
endif()

#
# Set ScalFMM compile flags
# -------------------------
set(${CMAKE_PROJECT_NAME}_CXX_FLAGS "${${CMAKE_PROJECT_NAME}_CXX_FLAGS}" CACHE STRING "Global compile flags for ScalFMM")
message(STATUS "ScalFMM final flags : ${${CMAKE_PROJECT_NAME}_CXX_FLAGS}")
