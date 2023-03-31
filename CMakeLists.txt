cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME neuralib)
project(${PROJECT_NAME} CXX)

set(CMAKE_CXX_STANDARD 17)
set(WARNINGS_AS_ERRORS OFF)
set(ENABLE_PVS_STUDIO OFF)
set(ENABLE_UBSan OFF)
set(ENABLE_ASAN OFF)
set(ENABLE_TSan OFF)
set(ENABLE_MSAN OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -Wno-deprecated-declarations -DNDEBUG")

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif ()

#include(cmake/CompilerWarnings.cmake)

add_executable(${PROJECT_NAME} main.cpp)

target_compile_options(${PROJECT_NAME} PUBLIC -O3)
#
#! Put path to your project headers

#! Add external packages
find_package(Boost 1.71.0 COMPONENTS program_options system REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} Boost::program_options Boost::system)

#target_link_libraries(${PROJECT_NAME} blas)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (neuralib Eigen3::Eigen)

INSTALL(PROGRAMS
		$<TARGET_FILE:${PROJECT_NAME}>
		DESTINATION bin)

set(ALL_TARGETS ${PROJECT_NAME})

include(cmake/main-config.cmake)