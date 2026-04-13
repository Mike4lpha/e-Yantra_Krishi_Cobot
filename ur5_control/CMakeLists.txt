cmake_minimum_required(VERSION 3.8)
project(ur5_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python scripts
install(PROGRAMS
    src/task1b_boiler_plate.py
    src/task1b.py
    src/task1c.py
    src/task2b_manipulation.py
    src/task2b_perception.py
    DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    src/bonus_task2_boiler_plate.py
    src/task3a.py
    src/task3b_manipulation.py
    src/task3b_perception.py
    src/task3b.py
    src/task4a_manipulation_sim.py
    src/task4a_manipulation.py
    src/task4a_perception.py
    src/task4c_perception.py
    src/task4c_manipulation.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
