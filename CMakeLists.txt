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

include(FetchContent)

# 1. Eigen (Linear Algebra)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.1
  FIND_PACKAGE_ARGS NAMES Eigen3
)
FetchContent_MakeAvailable(Eigen3)
#Find the Eigen3 package installed via Homebrew
#find_package(Eigen3 REQUIRED)

# 2. GoogleTest (Unit Testing)
FetchContent_Declare(
  GTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.18.0
  FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(GTest)
# Find the locally installed Google Test package from Homebrew
#find_package(GTest REQUIRED)
set(BOOST_INCLUDE_LIBRARIES math)
set(BOOST_ENABLE_CMAKE ON)
set(BOOST_ENABLE_COMPATIBILITY_TARGETS ON)
include(FetchContent)
FetchContent_Declare(
    Boost
	URL https://github.com
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	FIND_PACKAGE_ARGS NAMES Boost
)
FetchContent_MakeAvailable(Boost)

# Core Library
add_library(pricer_core
    src/Process.cpp
    src/MonteCarloEngine.cpp
)
target_include_directories(pricer_core PUBLIC include)
target_link_libraries(pricer_core PUBLIC Eigen3::Eigen Boost::headers)

# Unit Tests
enable_testing()
add_executable(run_tests tests/test_margrabe.cpp
	tests/test_autocall.cpp
)
target_link_libraries(run_tests PRIVATE pricer_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)
