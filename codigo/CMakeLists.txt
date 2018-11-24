cmake_minimum_required(VERSION 3.9.4)
project(tp3)

set(CMAKE_CXX_FLAGS -g)
set(CMAKE_CXX_FLAGS -O2)
set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Leemos todos los archivos fuentes en ./src
file(GLOB SOURCE_FILES src/ src/*.h)

# Leemos todos los archivos de test en ./tests
file(GLOB TEST_SOURCES tests/*.cpp)

add_subdirectory(tests/google-test)

# Creamos el ejecutable para correr los tests
add_executable(correrTests ${TEST_SOURCES} ${SOURCE_FILES} src/imageHandling.h)

add_executable(tp3 src/main.cpp ${SOURCE_FILES} src/imageHandling.h)

# add_executable(power_method_test src/power_method_test.cpp src/PCA.h)

if (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
endif (APPLE)

# Necesitamos asociar los archivos del framework de testing
target_link_libraries(correrTests gtest gtest_main)

#Asociamos los archivos de la biblioteca OpenCV
target_link_libraries( correrTests ${OpenCV_LIBS} )
target_link_libraries( tp3 ${OpenCV_LIBS} )

add_test(correrTests correrTests)

# Target para correr Valgrind
add_custom_target(correrValgrind
        valgrind --leak-check=full -v ./correrTests 2>&1
        DEPENDS correrTests)
