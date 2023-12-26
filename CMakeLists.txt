cmake_minimum_required(VERSION 3.8)

project(PrettyTools_CPP VERSION 0.1.9 LANGUAGES CXX)
message(STATUS "=== ${PROJECT_NAME} Cmaking ===")

include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

# include(CTest)

# 一些要统一的临时名称, 确保会被unset
set(MAIN_TARGET_NAME pretty_tools)
set(cmake_config_version_name ${MAIN_TARGET_NAME}-configversion)
set(cmake_config_name ${MAIN_TARGET_NAME}-config)
set(cmake_target_name ${MAIN_TARGET_NAME}-targets)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${cmake_config_name}.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${cmake_config_version_name}.cmake"
    DESTINATION lib/cmake/${MAIN_TARGET_NAME}
)

# 临时安装路径
# set(CMAKE_INSTALL_PREFIX "/workspace/tmp")

# ! 同时构建 动态链接库 和 静态链接库
# ----------------------------------------------------------------
aux_source_directory(src/datastruct datastruct_src)
set(PRETTY_TOOLS_SRC ${PRETTY_TOOLS_SRC} ${datastruct_src})
set(PRETTY_TOOLS_INC_DIRS ${PRETTY_TOOLS_INC_DIRS} inc/datastruct)

# ----------------------------------------------------------------
message(STATUS "PRETTY_TOOLS_SRC = ${PRETTY_TOOLS_SRC}")
message(STATUS "PRETTY_TOOLS_INC_DIRS = ${PRETTY_TOOLS_INC_DIRS}")

add_library(${MAIN_TARGET_NAME}_shared SHARED ${PRETTY_TOOLS_SRC})
add_library(${MAIN_TARGET_NAME}_static STATIC ${PRETTY_TOOLS_SRC})
target_include_directories(${MAIN_TARGET_NAME}_shared PRIVATE ${PRETTY_TOOLS_INC_DIRS})
target_include_directories(${MAIN_TARGET_NAME}_static PUBLIC ${PRETTY_TOOLS_INC_DIRS})

# 将 pretty-tools-cpp_shared 重新命名为 pretty-tools-cpp.so
set_target_properties(${MAIN_TARGET_NAME}_shared PROPERTIES OUTPUT_NAME ${MAIN_TARGET_NAME})

# 将 pretty-tools-cpp_static  重新命名为 pretty-tools-cpp.a
set_target_properties(${MAIN_TARGET_NAME}_static PROPERTIES OUTPUT_NAME ${MAIN_TARGET_NAME})

# ! 安装部分
install(TARGETS ${MAIN_TARGET_NAME}_static
    LIBRARY DESTINATION lib/${MAIN_TARGET_NAME} # 动态库安装路径
    ARCHIVE DESTINATION lib/${MAIN_TARGET_NAME} # 静态库安装路径

    # RUNTIME DESTINATION bin # 可执行文件安装路径
    # PUBLIC_HEADER DESTINATION include # 头文件安装路径
)

install(TARGETS ${MAIN_TARGET_NAME}_shared
    EXPORT ${cmake_target_name}
    LIBRARY DESTINATION lib/${MAIN_TARGET_NAME} # 动态库安装路径
    ARCHIVE DESTINATION lib/${MAIN_TARGET_NAME} # 静态库安装路径

    # RUNTIME DESTINATION bin # 可执行文件安装路径
    # PUBLIC_HEADER DESTINATION include # 头文件安装路径
)
set(EXPORT_LIST_TARGETS_SHARED ${EXPORT_LIST_TARGETS_SHARED} ${CMAKE_INSTALL_PREFIX}/lib/${MAIN_TARGET_NAME}/lib${MAIN_TARGET_NAME}.so) # * 当前版本是将所有的实现文件都编译到一起，以后再考虑分开编译
set(EXPORT_LIST_TARGETS_STATIC ${EXPORT_LIST_TARGETS_STATIC} ${CMAKE_INSTALL_PREFIX}/lib/${MAIN_TARGET_NAME}/lib${MAIN_TARGET_NAME}.a) # * 当前版本是将所有的实现文件都编译到一起，以后再考虑分开编译

# 安装头文件
install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/
    DESTINATION include/${MAIN_TARGET_NAME})

# ================= 引用辅助工具 =================
include(CMakePackageConfigHelpers)

# 声明版本信息
write_basic_package_version_file(
    ${cmake_config_version_name}.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion # 表示该函数库向下兼容
)

# 安装find_package()所需要的配置文件
install(EXPORT ${cmake_target_name}
    FILE ${cmake_target_name}.cmake
    NAMESPACE ${MAIN_TARGET_NAME}::
    DESTINATION lib/cmake/${MAIN_TARGET_NAME}
)

set(PACKAGE_PRETTY_TOOLS_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include/${MAIN_TARGET_NAME})
set(PACKAGE_PRETTY_TOOLS_LIBS_DIRS ${CMAKE_INSTALL_PREFIX}/lib/${MAIN_TARGET_NAME})

# @ONLY：限制变量替换，让其只替换被@VAR@引用的变量(那么${VAR}格式的变量将不会被替换)。这在配置${VAR}语法的脚本时是非常有用的
configure_file(${cmake_config_name}.cmake.in ${cmake_config_name}.cmake @ONLY)

# ! ============================== YAML-CPP 部分 ==============================
message(STATUS "=== module YAML-CPP Cmaking ===")
option(YAML_BUILD_SHARED_LIBS "Build yaml-cpp shared library" ${YAML_BUILD_SHARED_LIBS}) # 如果要生成为 SHARED ，可以指定 -DYAML_BUILD_SHARED_LIBS=ON

set(YAML_CPP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${MAIN_TARGET_NAME}"
    CACHE STRING "Path to install the CMake package to")

if(YAML_BUILD_SHARED_LIBS)
    set(yaml-cpp-type SHARED)
    set(yaml-cpp-label-postfix "shared")
else()
    set(yaml-cpp-type STATIC)
    set(yaml-cpp-label-postfix "static")
endif()

set(contrib-pattern "src/yaml-cpp/contrib/*.cpp")
set(src-pattern "src/yaml-cpp/*.cpp")
file(GLOB yaml-cpp-contrib-sources ${contrib-pattern})
file(GLOB yaml-cpp-sources ${src-pattern})
add_library(yaml-cpp ${yaml-cpp-type} "")
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)

set_property(TARGET yaml-cpp
    PROPERTY
    CXX_STANDARD_REQUIRED ON)

target_include_directories(yaml-cpp
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/yaml-cpp>)

target_sources(yaml-cpp
    PRIVATE
    ${yaml-cpp-contrib-sources}
    ${yaml-cpp-sources})

set_target_properties(yaml-cpp
    PROPERTIES
    CXX_STANDARD 11)

install(TARGETS yaml-cpp
    EXPORT yaml-cpp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${MAIN_TARGET_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${MAIN_TARGET_NAME})

install(EXPORT yaml-cpp-targets
    NAMESPACE yaml-cpp::
    DESTINATION "${YAML_CPP_INSTALL_CMAKEDIR}")

# ---------------------  后续处理 -----------------------
unset(PACKAGE_PRETTY_TOOLS_INCLUDE_DIRS)
unset(PACKAGE_PRETTY_TOOLS_LIBS_DIRS)
unset(cmake_config_version_name)
unset(cmake_config_name)
unset(cmake_target_name)
