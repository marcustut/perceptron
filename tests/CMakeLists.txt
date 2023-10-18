enable_testing()
find_package(GTest REQUIRED)

file(GLOB SOURCES "*.cpp")

add_executable(${CMAKE_PROJECT_NAME}_test ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME}_test GTest::gtest_main ${CMAKE_PROJECT_NAME})
target_include_directories(${CMAKE_PROJECT_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(${CMAKE_PROJECT_NAME}_test)
