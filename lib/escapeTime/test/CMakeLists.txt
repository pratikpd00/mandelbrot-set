add_executable(EscapeTimeTests 
	escapeTimeTests.cpp
)
target_include_directories(EscapeTimeTests PUBLIC ..)
target_link_libraries(EscapeTimeTests gtest_main)
target_link_libraries(EscapeTimeTests escapeTime)

include(GoogleTest)
gtest_discover_tests(EscapeTimeTests)