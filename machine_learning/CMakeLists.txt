cmake_minimum_required(VERSION 3.22.1)

project(machine_learning)

add_subdirectory(common_utils)
add_subdirectory(regression)
add_subdirectory(testing)

# add_executable(main # test.cpp # test1.cpp src/utils.cpp linear_regression.cpp # quadratic_regression.cpp)
# target_include_directories(main PRIVATE include)
# target_link_libraries(main PRIVATE matplot)