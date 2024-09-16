add_library(escapeTime STATIC
	escapeTimeCuda.cu
	escapeTimeSequential.cpp
)

set_target_properties(escapeTime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_TESTING)
	add_subdirectory(test)
endif()

if(BENCHMARK) 
	add_subdirectory(benchmark)
endif()

#target_include_directories(escapeTime PUBLIC ${CMAKE_SOURCE_DIR}/include)