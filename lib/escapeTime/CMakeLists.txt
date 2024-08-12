add_library(escapeTime SHARED escapeTimeCUDA.cu  "escapeTimeCuda.h")

target_include_directories(escapeTime PUBLIC ../../include)