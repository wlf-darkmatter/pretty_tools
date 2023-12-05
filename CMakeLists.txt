cmake_minimum_required(VERSION 3.8)

project(PrettyTools_CPP VERSION 0.1.9 LANGUAGES CXX)

set(PrettyTools_CPP_MAIN_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(PrettyTools_CPP_MAIN_PROJECT ON)
endif()

include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CTest)

option(PrettyTools_CPP_INSTALL "Enable generation of pretty_tools-cpp install targets" ${PrettyTools_CPP_MAIN_PROJECT})


cmake_dependent_option(PrettyTools_MSVC_SHARED_RT
  "MSVC: Build pretty_tools-cpp with shared runtime libs (/MD)" ON
  "CMAKE_SYSTEM_NAME MATCHES Windows" OFF)

set(PrettyTools_CPP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/pretty_tools"
  CACHE STRING "Path to install the CMake package to")

if (PrettyTools_BUILD_SHARED_LIBS)
  set(pretty-tools-cpp-type SHARED)
  set(pretty-tools-cpp-label-postfix "shared")
else()
  set(pretty-tools-cpp-type STATIC)
  set(pretty-tools-cpp-label-postfix "static")
endif()

#! 构建动态链接库
add_library(pretty-tools-cpp ${pretty-tools-cpp-type} "")

aux_source_directory(./src/datastruct PRETTY_TOOLS_CPP_SRC)
message(STATUS "PRETTY_TOOLS_CPP_SRC = "${PRETTY_TOOLS_CPP_SRC})

#! 添加头文件
include_directories(./inc/datastruct)

set(build-shared $<BOOL:${PrettyTools_BUILD_SHARED_LIBS}>)
set(build-windows-dll $<AND:$<BOOL:${CMAKE_HOST_WIN32}>,${build-shared}>)
set(not-msvc $<NOT:$<CXX_COMPILER_ID:MSVC>>)
set(msvc-shared_rt $<BOOL:${PrettyTools_MSVC_SHARED_RT}>)


target_sources(pretty-tools-cpp
    PRIVATE
        ${PRETTY_TOOLS_CPP_SRC})





