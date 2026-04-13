cmake_minimum_required(VERSION 3.8)
project(ebot_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

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

install(PROGRAMS
    src/ebot_nav_task1A.py
    src/ebot_nav_task2a.py
    src/shape_detector_task2a.py
    src/ebot_nav_task3b.py
    src/shape_detector_task3b.py
    src/ebot_nav_task4b.py
    src/shape_detector_task4b.py
    src/ebot_nav_task4c.py
    src/shape_detector_task4c.py
    src/ebot_nav_task4b_sim.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
