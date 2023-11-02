cmake_minimum_required(VERSION 3.22)

project(rhubarb CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_STANDARD 99)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

message(${CMAKE_CXX_COMPILER_ID})

set(CMAKE_CXX_FLAGS
    "-O${OPTIMIZATION_LEVEL} -fopenmp -fopenmp-simd -DLIKWID_PERFMON -DTBB_ENABLE_IPO=OFF -DBSIZE=${BSIZE} -DBWIDTH=${BWIDTH} -DCOMPUTE_ERROR=${COMPUTE_ERROR}"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -mavx -w -march=native -m64 -ftree-vectorize")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -pedantic")

set(profile FALSE)

if(profile)
  set(profile_dir ${CMAKE_CURRENT_SOURCE_DIR}/profile/rhb)
  set(profile_flag -fprofile-use) # generate/use

  add_link_options(${profile_flag}=${profile_dir})
  add_compile_options(${profile_flag}=${profile_dir} ${compile_options})
endif()

# output optimization remarks add_compile_options(-Wall -Wextra -Wpedantic
# -fsave-optimization-record)

find_package(likwid REQUIRED)
find_package(fmt REQUIRED)
include_directories(${likwid_INCLUDE_DIRS})

FetchContent_Declare(
  spray
  GIT_REPOSITORY https://github.com/atrostan/spray.git
  GIT_TAG master)

FetchContent_GetProperties(spray)
if(NOT spray_POPULATED)
  FetchContent_Populate(spray)
endif()

message(spray_SOURCE_DIR="${spray_SOURCE_DIR}")

FetchContent_MakeAvailable(spray)

find_package(oneDPL REQUIRED)
# BSIZE and BWIDTH are two compile time constants needed for spray
message(
  STATUS
    "Compiling Rhubarb with DOUBLE_SCORE_T = ${DOUBLE_SCORE_T}, COMPUTE_ERROR = ${COMPUTE_ERROR}, Spray with BSIZE = ${BSIZE}, BWIDTH = ${BWIDTH}"
)

set(RHUBARB_HEADERS include/io.h include/util.h include/CSR.h include/Rhubarb.h
                    include/likwid_defines.h include/EdgeListReader.h)
# set(APP_HEADERS apps/pr.h apps/cc.h apps/cf.h)
set(APP_HEADERS apps/pr.h)
set(UTIL_SRC src/io.cpp src/util.cpp)

add_executable(
  rhubarb_main src/rhubarb_main.cpp ${UTIL_SRC}
               ${RHUBARB_HEADERS} ${APP_HEADERS} include/platform_atomics.h)
add_executable(translate src/translate.cpp ${UTIL_SRC} ${RHUBARB_HEADERS})
add_executable(compress src/compress.cpp ${UTIL_SRC} ${RHUBARB_HEADERS})

add_library(spray INTERFACE)

target_include_directories(spray INTERFACE ${spray_SOURCE_DIR}/include)
target_include_directories(rhubarb_main PRIVATE include/)
target_include_directories(translate PRIVATE include/)
target_include_directories(compress PRIVATE include/)

# foreach(exec IN LISTS compress translate rhubarb_main)
foreach(exec IN LISTS rhubarb_main)
  target_compile_definitions(${exec} PUBLIC BSIZE=${BSIZE})
  target_compile_definitions(${exec} PUBLIC BWIDTH=${BWIDTH})
  target_compile_definitions(${exec} PUBLIC DOUBLE_SCORE_T=${DOUBLE_SCORE_T})
  target_compile_definitions(${exec} PUBLIC COMPUTE_ERROR=${COMPUTE_ERROR})
  target_compile_definitions(${exec} PUBLIC WEIGHTED_EDGE=${WEIGHTED_EDGE})
endforeach()
target_compile_definitions(translate PUBLIC BSIZE=${BSIZE})
target_link_libraries(rhubarb_main PUBLIC fmt::fmt oneDPL likwid spray)
target_link_libraries(compress PUBLIC fmt::fmt oneDPL likwid spray)
target_link_libraries(translate PUBLIC fmt::fmt oneDPL likwid spray)
