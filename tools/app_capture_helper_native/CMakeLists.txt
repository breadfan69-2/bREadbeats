cmake_minimum_required(VERSION 3.20)
project(app_capture_helper_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(app_capture_helper_native
    src/main.cpp
    src/loopback_capture.cpp
)

target_include_directories(app_capture_helper_native PRIVATE src)

target_compile_definitions(app_capture_helper_native PRIVATE UNICODE _UNICODE)

if(MSVC)
    target_compile_options(app_capture_helper_native PRIVATE /W4 /permissive-)
else()
    target_compile_options(app_capture_helper_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Core Audio / COM libraries needed for real implementation.
target_link_libraries(app_capture_helper_native PRIVATE
    ole32
    avrt
)
