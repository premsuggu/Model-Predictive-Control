cmake_minimum_required(VERSION 3.10)
project(NMPC_SIM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CasADi root folder (adjust as needed)
set(CASADI_DIR "${CMAKE_SOURCE_DIR}")

# Include CasADi headers (FIXED)
include_directories(
    "${CASADI_DIR}/casadi"
    "${CASADI_DIR}/casadi/build"
)

# Main executable
add_executable(nlmpc main.cpp nmpc_utils.cpp)

# Link CasADi library (adjust for your platform)
target_link_libraries(nlmpc "${CASADI_DIR}/casadi/build/Release/casadi.lib")
