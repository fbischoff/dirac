cmake_minimum_required(VERSION 3.23)
project(dirac_hatom)

set(CMAKE_CXX_STANDARD 17)
find_package(MADNESS CONFIG REQUIRED)
#find_package(LIBXC CONFIG REQUIRED)

add_executable(dirac_hatom main.cpp)
target_include_directories(dirac_hatom PUBLIC /opt/local/include)
target_link_libraries(dirac_hatom madness)
