add_library(escapeTime STATIC
	escapeTimeCuda.cu
	escapeTimeSequential.cpp
)

set_target_properties(escapeTime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(test)

target_include_directories(escapeTime PUBLIC ${CMAKE_SOURCE_DIR}/include)