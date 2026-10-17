cmake_minimum_required(VERSION 3.16)
project(linc VERSION 0.1.0 LANGUAGES C)

option(LINC_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(LINC_BUILD_BENCH "Build the benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(LINC_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(LINC_NATIVE "Tune for the build machine (-march/-mcpu=native)" OFF)
option(LINC_BENCH_BLAS "Benchmark against the system BLAS" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(linc
  src/linc_blas.c
  src/linc_chol.c
  src/linc_core.c
  src/linc_eig.c
  src/linc_lu.c
  src/linc_qr.c)
add_library(linc::linc ALIAS linc)

set_target_properties(linc PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(linc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

find_library(LINC_LIBM m)
if(LINC_LIBM)
  target_link_libraries(linc PUBLIC ${LINC_LIBM})
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(linc PRIVATE -Wall -Wextra -Wpedantic -Wshadow
    -Wstrict-prototypes -Wmissing-prototypes -Wcast-qual -Wvla
    -ffp-contract=fast)
  if(LINC_NATIVE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      target_compile_options(linc PRIVATE -mcpu=native)
    else()
      target_compile_options(linc PRIVATE -march=native)
    endif()
  endif()
elseif(MSVC)
  target_compile_options(linc PRIVATE /W4)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS linc EXPORT lincTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/linc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT lincTargets NAMESPACE linc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/linc)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lincConfig.cmake
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/lincTargets.cmake\")\n")
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/lincConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lincConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/lincConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/linc)

if(LINC_BUILD_TESTS)
  enable_testing()
  add_executable(test_linc tests/test_linc.c)
  target_link_libraries(test_linc PRIVATE linc)
  add_test(NAME linc_tests COMMAND test_linc)
endif()

if(LINC_BUILD_BENCH)
  add_executable(bench_linc bench/bench.c)
  target_link_libraries(bench_linc PRIVATE linc)
  if(LINC_BENCH_BLAS)
    if(APPLE)
      target_compile_definitions(bench_linc PRIVATE
        LINC_BENCH_CBLAS ACCELERATE_NEW_LAPACK)
      target_link_libraries(bench_linc PRIVATE "-framework Accelerate")
    else()
      find_package(BLAS REQUIRED)
      target_compile_definitions(bench_linc PRIVATE LINC_BENCH_CBLAS)
      target_link_libraries(bench_linc PRIVATE BLAS::BLAS)
    endif()
  endif()
endif()

if(LINC_BUILD_EXAMPLES)
  foreach(ex solve_system least_squares pca)
    add_executable(${ex} examples/${ex}.c)
    target_link_libraries(${ex} PRIVATE linc)
  endforeach()
endif()
