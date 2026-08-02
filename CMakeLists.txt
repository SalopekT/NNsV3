cmake_minimum_required(VERSION 3.16)

project(EigenExample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(eigen_example
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
)

target_link_libraries(eigen_example
    PRIVATE Eigen3::Eigen
)