set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(GTEST_VERSION v1.15.2)

cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CUDA_ARCHITECTURES "native")
project ("mandelbrot" LANGUAGES CXX CUDA)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        ${GTEST_VERSION}
)

find_package(Qt6 REQUIRED COMPONENTS Core)

if(BENCHMARK)
	add_compile_definitions(BENCHMARKING)
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
find_package(Qt6 REQUIRED COMPONENTS 
	Widgets
)


include(CTest)
add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(src)


