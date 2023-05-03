cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME neuralib)
project(${PROJECT_NAME} CXX)

set(CMAKE_CXX_STANDARD 20)
set(ENABLE_UBSan OFF)
set(ENABLE_ASAN OFF)
set(ENABLE_TSan OFF)
set(ENABLE_MSAN OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -Wno-deprecated-declarations -DNDEBUG -fsanitize=undefined")

find_program(NVCC nvcc)

if (NVCC)
	enable_language(CUDA)
endif ()

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif ()

# Uncomment this macros to use cuda
#add_definitions(-DCUDA_ENABLE)

add_executable(${PROJECT_NAME} main.cpp src/layers/Layer.cpp include/layers/Layer.h include/layers/Layer.h src/layers/Layer.cpp src/layers/Dense.cpp include/layers/Dense.h src/utils/Initializer.cpp include/utils/Initializer.h src/layers/Activation.cpp include/layers/Activation.h src/utils/Optimizer.cpp include/utils/Optimizer.h src/models/Model.cpp include/models/Model.h src/utils/Loss.cpp include/utils/Loss.h include/utils/Dataset.h include/utils/tqdm.h)

target_compile_options(${PROJECT_NAME} PUBLIC -O3)
#
#! Put path to your project headers
include_directories(include)
include_directories(src)
#! Add external packages
#find_package(Boost 1.71.0 COMPONENTS program_options system REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIR})
#target_link_libraries(${PROJECT_NAME} Boost::program_options Boost::system)
find_package (Eigen3 REQUIRED)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_package(TBB REQUIRED)
	target_include_directories(${PROJECT_NAME} PRIVATE ${TBB_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} TBB::tbb)
else()
	find_package(TBB REQUIRED)
	target_include_directories(${PROJECT_NAME} PRIVATE ${TBB_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${TBB_LIBRARIES} -ltbb)
endif ()

# Enabling cuda support
if(NVCC)
	set(CMAKE_CUDA_STANDARD 17)

	find_package(CUDA REQUIRED)

	set(CUDA_SOURCES include/utils/cuda/matrix_operations.cuh src/utils/cuda/matrix_operations.cu)

	cuda_add_library(my_cuda_lib ${CUDA_SOURCES})

	target_link_libraries(${PROJECT_NAME} my_cuda_lib)
	target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif()

INSTALL(PROGRAMS
		$<TARGET_FILE:${PROJECT_NAME}>
		DESTINATION bin)

set(ALL_TARGETS ${PROJECT_NAME})

include(cmake/main-config.cmake)
