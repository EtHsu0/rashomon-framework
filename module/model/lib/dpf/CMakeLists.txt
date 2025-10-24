# CMakeList.txt : CMake project for DPF, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("DPF")
include(CTest)
set (CMAKE_CXX_STANDARD 17)

# Set release build.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
	message(STATUS "Build type not specified: Use Release by default")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL Release)
	message(STATUS "Compiling in release mode")	
	add_compile_definitions(DEBUG=0)
else()
	message(STATUS "Compiling in debug mode")	
	add_compile_definitions(DEBUG=1)
endif()

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

set(DPF_HEADER_FILES
	${PROJECT_INCLUDE_DIR}/base.h

	${PROJECT_INCLUDE_DIR}/utils/parameter_handler.h
	${PROJECT_INCLUDE_DIR}/utils/stopwatch.h
	${PROJECT_INCLUDE_DIR}/utils/file_reader.h
	
	${PROJECT_INCLUDE_DIR}/model/feature_vector_binary.h
	${PROJECT_INCLUDE_DIR}/model/branch.h
	${PROJECT_INCLUDE_DIR}/model/binary_data.h
	${PROJECT_INCLUDE_DIR}/model/partial_solution.h
	${PROJECT_INCLUDE_DIR}/model/internal_node_description.h
	${PROJECT_INCLUDE_DIR}/model/pareto_front.h
	${PROJECT_INCLUDE_DIR}/model/container.h

	${PROJECT_INCLUDE_DIR}/solver/solver_result.h
	${PROJECT_INCLUDE_DIR}/solver/feature_selector.h
	${PROJECT_INCLUDE_DIR}/solver/difference_computer.h
	${PROJECT_INCLUDE_DIR}/solver/similarity_lowerbound.h
	${PROJECT_INCLUDE_DIR}/solver/abstract_cache.h
	${PROJECT_INCLUDE_DIR}/solver/dataset_cache.h
	${PROJECT_INCLUDE_DIR}/solver/branch_cache.h
	${PROJECT_INCLUDE_DIR}/solver/counter.h
	${PROJECT_INCLUDE_DIR}/solver/decision_node.h
	${PROJECT_INCLUDE_DIR}/solver/terminal_solver.h
	${PROJECT_INCLUDE_DIR}/solver/statistics.h
	${PROJECT_INCLUDE_DIR}/solver/solver.h
)

set(DPF_SRC_FILES

	${PROJECT_SOURCE_DIR}/utils/parameter_handler.cpp
	${PROJECT_SOURCE_DIR}/utils/file_reader.cpp

	${PROJECT_SOURCE_DIR}/model/feature_vector_binary.cpp
	${PROJECT_SOURCE_DIR}/model/branch.cpp
	${PROJECT_SOURCE_DIR}/model/binary_data.cpp
	${PROJECT_SOURCE_DIR}/model/partial_solution.cpp
	${PROJECT_SOURCE_DIR}/model/pareto_front.cpp


	${PROJECT_SOURCE_DIR}/solver/dataset_cache.cpp
	${PROJECT_SOURCE_DIR}/solver/branch_cache.cpp
	${PROJECT_SOURCE_DIR}/solver/feature_selector_gini.cpp
	${PROJECT_SOURCE_DIR}/solver/define_parameters.cpp
	${PROJECT_SOURCE_DIR}/solver/solver.cpp
	${PROJECT_SOURCE_DIR}/solver/counter.cpp
	${PROJECT_SOURCE_DIR}/solver/terminal_solver.cpp
	${PROJECT_SOURCE_DIR}/solver/frequency_counter.cpp
	${PROJECT_SOURCE_DIR}/solver/solver_result.cpp
	${PROJECT_SOURCE_DIR}/solver/similarity_lowerbound.cpp
	${PROJECT_SOURCE_DIR}/solver/difference_computer.cpp

	${DPF_HEADER_FILES}
)

# Add source to this project's executable.
add_library(DPFLib ${DPF_SRC_FILES})
add_executable (DPF ${PROJECT_SOURCE_DIR}/main.cpp)
add_executable (DPFBenchmark ${PROJECT_SOURCE_DIR}/benchmark.cpp)

target_include_directories(DPFLib PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries (DPF DPFLib)
target_link_libraries (DPFBenchmark DPFLib)

if (CMAKE_BUILD_TYPE STREQUAL Release)
	
else()
	if (MSVC)
		message(STATUS "Enabling profiler")	
		#set_target_properties(DPFLib DPF PROPERTIES LINK_FLAGS "/PROFILE")
		SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
	else ()
		message(STATUS "No profiler")
	endif()
endif()

# Tests

enable_testing()
set(PROJECT_TEST_DIR "test")
add_executable(ParetoTest ${PROJECT_TEST_DIR}/pareto_test.cpp)
target_link_libraries (ParetoTest DPFLib)
add_test(NAME "ParetoTest" COMMAND ParetoTest)