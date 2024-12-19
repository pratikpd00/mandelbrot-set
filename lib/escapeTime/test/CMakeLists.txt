add_executable(EscapeTimeTests 
	escapeTimeTests.cpp
	imageTransformGridTests.cpp
)
target_include_directories(EscapeTimeTests PUBLIC ${CMAKE_SOURCE_DIR}/include ..)
target_link_libraries(EscapeTimeTests gtest_main)
target_link_libraries(EscapeTimeTests escapeTime)

include(GoogleTest)
gtest_discover_tests(EscapeTimeTests)