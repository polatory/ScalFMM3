#
# FFT options
# -----------
find_path(POCKETFFT_INCLUDE_DIR pocketfft_hdronly.h REQUIRED)
add_library(pocketfft::pocketfft INTERFACE IMPORTED)
target_include_directories(pocketfft::pocketfft INTERFACE ${POCKETFFT_INCLUDE_DIR})
list(APPEND FFTW_TARGET pocketfft::pocketfft)
list(APPEND FUSE_LIST FFTW)
