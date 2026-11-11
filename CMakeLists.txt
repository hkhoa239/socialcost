cmake_minimum_required(VERSION 3.8)
project(socialcost)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(arena_social_cost_layer REQUIRED)

# Header-only use of arena_social_cost_layer/cost_math.hpp, no link to the plugin.
add_library(socialcost_grid SHARED src/socialcost_grid.cpp)
target_include_directories(socialcost_grid PRIVATE ${arena_social_cost_layer_INCLUDE_DIRS})
target_compile_options(socialcost_grid PRIVATE -O2 -Wall -Wextra)

install(TARGETS socialcost_grid LIBRARY DESTINATION lib)
ament_python_install_package(${PROJECT_NAME})
install(DIRECTORY scripts/ DESTINATION lib/${PROJECT_NAME} USE_SOURCE_PERMISSIONS)
install(DIRECTORY configs DESTINATION share/${PROJECT_NAME})

ament_package()
