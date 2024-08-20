add_executable(CUDAEscapeTimeTests 
	cudaEscapeTimeTests.cpp
)
target_include_directories(CUDAEscapeTimeTests PUBLIC ${CMAKE_SOURCE_DIR}/include /lib)
target_link_libraries(CUDAEscapeTimeTests gtest_main)
target_link_libraries(CUDAEscapeTimeTests escapeTime)

include(GoogleTest)
gtest_discover_tests(CUDAEscapeTimeTests)