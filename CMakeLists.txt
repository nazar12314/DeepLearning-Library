cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME neuralib)
project(${PROJECT_NAME} CXX)


set(CMAKE_CXX_STANDARD 20)
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

add_executable(${PROJECT_NAME} main.cpp src/layers/Layer.cpp include/layers/Layer.h include/layers/Layer.h src/layers/Layer.cpp src/layers/Dense.cpp include/layers/Dense.h src/utils/Initializer.cpp include/utils/Initializer.h src/layers/Activation.cpp include/layers/Activation.h src/utils/Optimizer.cpp include/utils/Optimizer.h src/models/Model.cpp include/models/Model.h src/utils/Loss.cpp include/utils/Loss.h src/utils/TensorHolder.cpp include/utils/TensorHolder.h)

target_compile_options(${PROJECT_NAME} PUBLIC -O3)
#
#! Put path to your project headers
include_directories(include)
include_directories(src)
#! Add external packages
find_package(Boost 1.71.0 COMPONENTS program_options system REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} Boost::program_options Boost::system)

find_package(MNIST PATHS mnist)
if(NOT MNIST_FOUND)
	message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)
target_compile_definitions(${PROJECT_NAME} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (neuralib Eigen3::Eigen)

INSTALL(PROGRAMS
		$<TARGET_FILE:${PROJECT_NAME}>
		DESTINATION bin)

set(ALL_TARGETS ${PROJECT_NAME})

include(cmake/main-config.cmake)