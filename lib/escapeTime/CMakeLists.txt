add_library(escapeTime SHARED escapeTimeCUDA.cu)

target_include_directories(escapeTime PUBLIC ../../include)