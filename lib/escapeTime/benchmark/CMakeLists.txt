add_executable(Benchmark
	benchmarkEscapeTime.cpp
)
target_include_directories(Benchmark PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(Benchmark escapeTime)