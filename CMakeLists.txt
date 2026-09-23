cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
project(EigenExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)

# MSVC debug flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")
endif()
set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

add_executable(eigen_example_cuda
    src/main.cpp
    src/Layers/Layer.cpp
    src/Layers/LinearLayer.cpp
    src/Network.cpp
    src/Activations/Relu.cpp
    src/Activations/Softmax.cpp
    src/Losses/CrossEntropyLoss.cpp
    src/MNISTdata/FileReader.cpp
    src/Layers/ConvolutionalLayer.cpp
    src/Layers/MCConvolutionalLayer.cpp
    src/Layers/MaxPooling.cpp
    src/Layers/MCMaxPooling.cpp
)

target_link_libraries(eigen_example_cuda
    PRIVATE Eigen3::Eigen
    PRIVATE CUDA::cudart
)

target_compile_definitions(eigen_example_cuda PRIVATE EIGEN_USE_CUDA)