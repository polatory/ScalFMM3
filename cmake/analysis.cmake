#
# Static analysis during build
# ----------------------------

if(${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS AND ${CMAKE_PROJECT_NAME}_USE_CPPCHECK)
  find_program(CPPCHECK "cppcheck")
  if (CPPCHECK)
    set(CMAKE_CXX_CPPCHECK "${CPPCHECK}"
      "--language=c++"
      "$<GENEX_EVAL:-I$<TARGET_PROPERTY:INTERFACE_INCLUDE_DIRECTORIES>>"
      "-I."
      "-v"
      "--enable=all"
      "--force"
      "--inline-suppr"
      )
    message(STATUS "CPPCHECK analysis is ON.")
  endif()
  #add_custom_target(genexdebug COMMAND ${CMAKE_COMMAND} -E echo $<TARGET_GENEX_EVAL:${CMAKE_PROJECT_NAME},-I$<TARGET_PROPERTY:${CMAKE_PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>>)
endif(${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS AND ${CMAKE_PROJECT_NAME}_USE_CPPCHECK)

if(${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS AND ${CMAKE_PROJECT_NAME}_USE_CLANGTIDY)
  find_program(CLANGTIDY "clang-tidy")
  if(CLANGTIDY)
    set(CMAKE_CXX_CLANG_TIDY "${CLANGTIDY};--header-filter=${CMAKE_SOURCE_DIR}/include/scalfmm/.;--checks=-*,cppcoreguidelines-*,clang-analyser-cplusplus*,modernize-*,mpi-*,performance-*,portability-*,readability-*")
    message(STATUS "Clang Tidy analysis is ON.")
  endif()
endif(${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS AND ${CMAKE_PROJECT_NAME}_USE_CLANGTIDY)

if(${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS AND ${CMAKE_PROJECT_NAME}_USE_IWYU)
  find_program(IWYU "include-what-you-use")
  if(IWYU)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU}")
    message(STATUS "Include What You Use analysis is ON.")
  endif()
endif(${CMAKE_PROJECT_NAME}_USE_STATIC_ANALYSIS AND ${CMAKE_PROJECT_NAME}_USE_IWYU)
