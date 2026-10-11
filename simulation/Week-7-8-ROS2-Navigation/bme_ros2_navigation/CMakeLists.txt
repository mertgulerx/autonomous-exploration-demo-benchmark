cmake_minimum_required(VERSION 3.8)
project(bme_ros2_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


set(_share_dirs
  launch
  worlds
  rviz
  urdf
  meshes
  config
  maps
)

set(_existing_share_dirs "")
foreach(_dir IN LISTS _share_dirs)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}")
    list(APPEND _existing_share_dirs "${_dir}")
  endif()
endforeach()

if(_existing_share_dirs)
  install(DIRECTORY
    ${_existing_share_dirs}
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_package()
