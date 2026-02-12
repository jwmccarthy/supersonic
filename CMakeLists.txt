cmake_minimum_required(VERSION 3.20)

project(Supersonic LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

file(GLOB SOURCES src/*.cpp src/*.cu)
add_executable(main ${SOURCES})

target_compile_features(main PRIVATE cxx_std_17)
set_target_properties(main PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost_pfr/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(main PRIVATE CUDA::cudart)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)
