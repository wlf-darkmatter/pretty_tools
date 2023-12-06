cmake_minimum_required(VERSION 3.8)

project(PrettyTools_CPP VERSION 0.1.9 LANGUAGES CXX)
message(STATUS "=== ${PROJECT_NAME} Cmaking ===")

include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

# include(CTest)

# 一些要统一的临时名称, 确保会被unset
set(TARGET_NAME pretty_tools)
set(cmake_config_version_name ${TARGET_NAME}-configversion)
set(cmake_config_name ${TARGET_NAME}-config)
set(cmake_target_name ${TARGET_NAME}-targets)

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

add_library(${TARGET_NAME}_shared SHARED ${PRETTY_TOOLS_SRC})
add_library(${TARGET_NAME}_static STATIC ${PRETTY_TOOLS_SRC})
target_include_directories(${TARGET_NAME}_shared PRIVATE ${PRETTY_TOOLS_INC_DIRS})
target_include_directories(${TARGET_NAME}_static PUBLIC ${PRETTY_TOOLS_INC_DIRS})

# 将 pretty-tools-cpp_shared 重新命名为 pretty-tools-cpp.so
set_target_properties(${TARGET_NAME}_shared PROPERTIES OUTPUT_NAME ${TARGET_NAME})

# 将 pretty-tools-cpp_static  重新命名为 pretty-tools-cpp.a
set_target_properties(${TARGET_NAME}_static PROPERTIES OUTPUT_NAME ${TARGET_NAME})

# ! 安装部分
install(TARGETS ${TARGET_NAME}_static
    LIBRARY DESTINATION lib/${TARGET_NAME} # 动态库安装路径
    ARCHIVE DESTINATION lib/${TARGET_NAME} # 静态库安装路径

    # RUNTIME DESTINATION bin # 可执行文件安装路径

    # PUBLIC_HEADER DESTINATION include # 头文件安装路径
)

install(TARGETS ${TARGET_NAME}_shared
    EXPORT ${cmake_target_name}
    LIBRARY DESTINATION lib/${TARGET_NAME} # 动态库安装路径
    ARCHIVE DESTINATION lib/${TARGET_NAME} # 静态库安装路径

    # RUNTIME DESTINATION bin # 可执行文件安装路径

    # PUBLIC_HEADER DESTINATION include # 头文件安装路径
)

# 安装头文件
install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/
    DESTINATION include/${TARGET_NAME})

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
    NAMESPACE ${TARGET_NAME}::
    DESTINATION lib/cmake/${TARGET_NAME}
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${cmake_config_name}.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${cmake_config_version_name}.cmake"
    DESTINATION lib/cmake/${TARGET_NAME}
)

set(PACKAGE_PRETTY_TOOLS_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include/${TARGET_NAME})
set(PACKAGE_PRETTY_TOOLS_LIBS_DIRS ${CMAKE_INSTALL_PREFIX}/lib/${TARGET_NAME})
set(EXPORT_TARGETS ${TARGET_NAME}) # * 当前版本是将所有的实现文件都编译到一起，以后再考虑分开编译

# @ONLY：限制变量替换，让其只替换被@VAR@引用的变量(那么${VAR}格式的变量将不会被替换)。这在配置${VAR}语法的脚本时是非常有用的
configure_file(${cmake_config_name}.cmake.in ${cmake_config_name}.cmake @ONLY)

unset(PACKAGE_PRETTY_TOOLS_INCLUDE_DIRS)
unset(PACKAGE_PRETTY_TOOLS_LIBS_DIRS)
unset(cmake_config_version_name)
unset(cmake_config_name)
unset(cmake_target_name)