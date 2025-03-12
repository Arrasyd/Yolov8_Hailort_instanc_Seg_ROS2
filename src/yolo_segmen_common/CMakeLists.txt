cmake_minimum_required(VERSION 3.8)
project(yolo_segmen_common)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)


ament_auto_add_library(${PROJECT_NAME} SHARED
  src/yolov8seg_example.cpp
  src/yolov8seg_postprocess.cpp
)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/include/yolo_segmen_common/)

ExternalProject_Add(xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

ExternalProject_Add(xtensor-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/include/yolo_segmen_common/
)

ExternalProject_Add(xtensor-blas-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/include/yolo_segmen_common/
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${EXTERNAL_INSTALL_LOCATION}/include
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

link_libraries(stdc++fs)
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test xtensor-blas-test)
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort ${OpenCV_LIBS} stdc++fs)
target_include_directories(${PROJECT_NAME} PUBLIC include ${OpenCV_INCLUDE_DIRS})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()