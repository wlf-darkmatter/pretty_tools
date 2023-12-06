cmake_minimum_required(VERSION 3.8)

project(PrettyTools_CPP VERSION 0.1.9 LANGUAGES CXX)
message(STATUS "=== PrettyTools_CPP Cmaking ===")

set(PrettyTools_CPP_MAIN_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(PrettyTools_CPP_MAIN_PROJECT ON)
endif()


include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CTest)

cmake_dependent_option(PrettyTools_MSVC_SHARED_RT
  "MSVC: Build pretty_tools-cpp with shared runtime libs (/MD)" ON
  "CMAKE_SYSTEM_NAME MATCHES Windows" OFF)

set(PrettyTools_CPP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/pretty_tools"
  CACHE STRING "Path to install the CMake package to")

# 临时安装路径
# set(CMAKE_INSTALL_PREFIX "/workspace/tmp")

#! 同时构建 动态链接库 和 静态链接库
#----------------------------------------------------------------
aux_source_directory(./src/datastruct datastruct_src)
set(PRETTY_TOOLS_SRC ${PRETTY_TOOLS_SRC} ${datastruct_src})
set(PRETTY_TOOLS_INC_DIRS ${PRETTY_TOOLS_INC_DIRS} ./inc/datastruct)

#一些要统一的临时名称, 确保会被unset
set(config_version_name pretty_tools-configversion)
set(config_name pretty_tools-config)
set(target_name pretty_tools-targets)

#----------------------------------------------------------------

message(STATUS "PRETTY_TOOLS_SRC = ${PRETTY_TOOLS_SRC}")
message(STATUS "PRETTY_TOOLS_INC_DIRS = ${PRETTY_TOOLS_INC_DIRS}")

add_library(pretty-tools-cpp_shared SHARED ${PRETTY_TOOLS_SRC})
add_library(pretty-tools-cpp_static STATIC ${PRETTY_TOOLS_SRC})
target_include_directories(pretty-tools-cpp_shared PRIVATE ${PRETTY_TOOLS_INC_DIRS})
target_include_directories(pretty-tools-cpp_static PUBLIC ${PRETTY_TOOLS_INC_DIRS})

# 将 pretty-tools-cpp_shared 重新命名为 pretty-tools-cpp.so
set_target_properties (pretty-tools-cpp_shared PROPERTIES OUTPUT_NAME "pretty-tools-cpp")
# 将 pretty-tools-cpp_static  重新命名为 pretty-tools-cpp.a
set_target_properties (pretty-tools-cpp_static PROPERTIES OUTPUT_NAME "pretty-tools-cpp")




#! 安装部分
install(TARGETS pretty-tools-cpp_static
        LIBRARY DESTINATION lib  # 动态库安装路径
        ARCHIVE DESTINATION lib  # 静态库安装路径
        RUNTIME DESTINATION bin  # 可执行文件安装路径
        PUBLIC_HEADER DESTINATION include  # 头文件安装路径
        )

install(TARGETS pretty-tools-cpp_shared
        EXPORT ${target_name}
        LIBRARY DESTINATION lib  # 动态库安装路径
        ARCHIVE DESTINATION lib  # 静态库安装路径
        RUNTIME DESTINATION bin  # 可执行文件安装路径
        PUBLIC_HEADER DESTINATION include  # 头文件安装路径
        )

set(EXPORT_TARGETS pretty-tools-cpp) #* 用于导出cmake变量

# 安装头文件
install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/
        DESTINATION include/pretty-tools)

# ================= 引用辅助工具 =================
include(CMakePackageConfigHelpers)

# 声明版本信息
write_basic_package_version_file(
    ${config_version_name}.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion  # 表示该函数库向下兼容
    )
# 安装find_package()所需要的配置文件
install(EXPORT ${target_name}
        FILE ${target_name}.cmake
        NAMESPACE pretty_tools::
        DESTINATION lib/cmake/pretty_tools
        )

configure_file(${config_name}.cmake.in ${config_name}.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${config_name}.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${config_version_name}.cmake"
        DESTINATION lib/cmake/pretty_tools
        )

unset(config_version_name)
unset(config_name)
unset(target_name)