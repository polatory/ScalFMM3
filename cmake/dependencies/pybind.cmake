#
# Python
# ---------

if(${CMAKE_PROJECT_NAME}_BUILD_PYFMM)
  #
  # Python 3.10 or newer ()
  # --------------------
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

  if(PYTHON_FOUND AND Python_NumPy_FOUND)
  # Print out the discovered paths
  cmake_print_variables(Python_VERSION)

  # cmake_print_variables(Python_INTERPRETER_ID)
  # cmake_print_variables(Python_EXECUTABLE)
  # cmake_print_variables(Python_INCLUDE_DIRS)
  # cmake_print_variables(F2PY_INCLUDE_DIR)
    cmake_print_variables(Python_NumPy_VERSION)
  # cmake_print_variables(Python_NumPy_INCLUDE_DIRS)
  # cmake_print_variables(Python_LIBRARIES)
  # cmake_print_variables(Python_ROOT_DIR)
     cmake_print_variables(Python_NumPy_FOUND)

     find_package(pybind11 REQUIRED)

     if(pybind11_FOUND)
         MESSAGE(STATUS "pybind11 Found")
        cmake_print_variables(pybind11_VERSION)
     else(pybind11_FOUND)
         MESSAGE(FATAL_ERROR "pybind11 NOT FOUND")
     endif(pybind11_FOUND)
  else()
    cmake_print_variables(PYTHON_FOUND)
    cmake_print_variables(Python_NumPy_FOUND)
    MESSAGE(FATAL_ERROR "Python or Numpy not found")
  endif()

endif()