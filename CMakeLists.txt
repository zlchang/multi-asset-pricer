cmake_minimum_required(VERSION 4.4)
project(MultiAssetPricer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable aggressive compiler optimizations for Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

#include(FetchContent)

# 1. Eigen (Linear Algebra)
#FetchContent_Declare(
#  Eigen
#  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#  GIT_TAG 3.4.0
#)
#FetchContent_MakeAvailable(Eigen)
#Find the Eigen3 package installed via Homebrew
find_package(Eigen3 REQUIRED)

# 2. GoogleTest (Unit Testing)
#FetchContent_Declare(
  #googletest
  #GIT_REPOSITORY https://github.com/google/googletest.git
  #GIT_TAG v1.14.0
#)
#FetchContent_MakeAvailable(googletest)
# Find the locally installed Google Test package from Homebrew
find_package(GTest REQUIRED)

# Core Library
add_library(pricer_core
    src/Process.cpp
    src/MonteCarloEngine.cpp
)
target_include_directories(pricer_core PUBLIC include)
target_link_libraries(pricer_core PUBLIC Eigen3::Eigen)

# Unit Tests
enable_testing()
add_executable(run_tests tests/test_margrabe.cpp)
target_link_libraries(run_tests PRIVATE pricer_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)
