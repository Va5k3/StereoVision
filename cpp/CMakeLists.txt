cmake_minimum_required(VERSION 3.20)
project(stereovision_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT OpenCV_DIR AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    set(_opencv_dir "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/opencv4")
    if(EXISTS "${_opencv_dir}/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_opencv_dir}" CACHE PATH "OpenCV from vcpkg")
    endif()
endif()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs calib3d highgui)

add_executable(stereovision
    src/main.cpp
    src/StereoVision.cpp
)
target_include_directories(stereovision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(stereovision PRIVATE ${OpenCV_LIBS})
