cmake_minimum_required(VERSION 3.5)
project(solver)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    main.cpp
    MoveSorter.hpp
    position.hpp
    solver.cpp
    solver.hpp
    TranspositionTable.hpp)

add_executable(solver ${SOURCE_FILES})