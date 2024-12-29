########
# Core #
########
### Use VC Prebuilt Libs ###
# Built using vc-deps
option(VC_PREBUILT_LIBS "Link against prebuilt dependencies" off)
if (VC_PREBUILT_LIBS)
    list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/vc-deps/deps)
endif()

### Filesystem ###
find_package(Filesystem)
if(Filesystem_FOUND)
    option(VC_USE_BOOSTFS "Use Boost as the filesystem library" OFF)
else()
    option(VC_USE_BOOSTFS "Use Boost as the filesystem library" ON)
endif()

if(VC_USE_BOOSTFS)
    add_compile_definitions(VC_USE_BOOSTFS)
    find_package(Boost 1.58 REQUIRED COMPONENTS system filesystem)
    set(VC_FS_LIB Boost::filesystem)
else()
    set(VC_FS_LIB std::filesystem)
endif()
message(STATUS "Using filesystem library: ${VC_FS_LIB}")
list(APPEND VC_CUSTOM_MODULES "${CMAKE_MODULE_PATH}/FindFilesystem.cmake")

### Qt6 ###
if((VC_BUILD_APPS OR VC_BUILD_UTILS) AND VC_BUILD_GUI)
    find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network)
    # qt_standard_project_setup() #NOTE below settings for QT < 6.3, commented command for qt >= 6.3, ubuntu 22.04 has qt 6.2!
     set(CMAKE_AUTOMOC ON)
     set(CMAKE_AUTORCC ON)
     set(CMAKE_AUTOUIC ON)
     
     if(NOT DEFINED qt_generate_deploy_app_script)
            message(WARNING "WARNING qt_generate_deploy_app_script MISSING!")
        function(qt_generate_deploy_app_script)
        endfunction()
     endif()
     
endif()

### ITK ###
set(VC_ITK_COMPONENTS
    ITKCommon
    ITKMesh
    ITKQuadEdgeMesh
    ITKTransform
    ITKIOTransformBase
    ITKIOTransformInsightLegacy
)
find_package(ITK 4.10 COMPONENTS ${VC_ITK_COMPONENTS} QUIET REQUIRED)
include(${ITK_USE_FILE})

### VTK ###
find_package(VTK 9 QUIET REQUIRED)

option(VC_WITH_CUDA_SPARSE "use cudss" ON)
if (VC_WITH_CUDA_SPARSE)
    add_definitions(-DVC_USE_CUDA_SPARSE=1)
endif()

#ceres-solver
find_package(Ceres REQUIRED)

### ACVD ###
include(BuildACVD)

### Z5 ###
include(BuildZ5)

### Eigen ###
find_package(Eigen3 3.3 REQUIRED)
if(CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
    message(AUTHOR_WARNING "Configuring a Debug build. Eigen performance will be degraded. If you need debug symbols, \
    consider setting CMAKE_BUILD_TYPE to RelWithDebInfo. Otherwise, set to Release to maximize performance.")
endif()

### OpenCV ###
find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4 QUIET REQUIRED)
endif()

set(XTENSOR_USE_XSIMD 1)
set(XTENSOR_USE_OPENMP 1)
find_package(xtensor REQUIRED)

find_package(OpenMP REQUIRED)

### libtiff ###
find_package(TIFF 4.0 REQUIRED)

### spdlog ###
find_package(spdlog 1.4.2 CONFIG REQUIRED)

### OpenABF ###
include(BuildOpenABF)

### Modern JSON ###
include(BuildJSON)

### bvh ###
include(Buildbvh)

### smgl ###
include(Buildsmgl)

### Boost and indicators (for app use only)
if(VC_BUILD_APPS OR VC_BUILD_UTILS)
    find_package(Boost 1.58 REQUIRED COMPONENTS system program_options)
    include(BuildIndicators)
endif()


############
# Optional #
############

### gtest ###
if(VC_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        CMAKE_CACHE_ARGS
            -DINSTALL_GTEST:BOOL=OFF
    )

    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        set(INSTALL_GTEST OFF CACHE BOOL OFF FORCE)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()

# Python bindings
if(VC_BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
endif()

### macOS security framework ###
if(APPLE AND VC_BUILD_APPS AND VC_BUILD_GUI)
    find_library(OSXSecurity Security)
endif()
