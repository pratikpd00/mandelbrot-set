add_library(escapeTime SHARED escapeTimeCUDA.cu)
add_subdirectory("test")

target_include_directories(escapeTime PUBLIC ../../include)