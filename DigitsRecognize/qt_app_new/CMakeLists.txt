cmake_minimum_required(VERSION 3.16)
project(DigitsRecognize VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- ML source folder ---
set(ML_SRC_DIR "${CMAKE_SOURCE_DIR}/../../src")

# --- GUI sources ---
set(GUI_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    Canvas.cpp
    Canvas.h
)

# --- ML sources ---
set(ML_SOURCES
    ${ML_SRC_DIR}/Network.cpp
    ${ML_SRC_DIR}/Layer.cpp
    ${ML_SRC_DIR}/LinearLayer.cpp
    ${ML_SRC_DIR}/Activations/Relu.cpp
    ${ML_SRC_DIR}/Activations/Softmax.cpp
    ${ML_SRC_DIR}/MNISTdata/FileReader.cpp
)

# --- Create executable ---
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(DigitsRecognize
        MANUAL_FINALIZATION
        ${GUI_SOURCES}
        ${ML_SOURCES}
    )
else()
    add_executable(DigitsRecognize ${GUI_SOURCES} ${ML_SOURCES})
endif()

# --- Include ML headers ---
target_include_directories(DigitsRecognize
    PRIVATE
    ${ML_SRC_DIR}
)

# --- Link libraries ---
target_link_libraries(DigitsRecognize
    PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Eigen3::Eigen
)

# --- Windows / macOS bundle settings ---
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.DigitsRecognize)
endif()

set_target_properties(DigitsRecognize PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS DigitsRecognize
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(DigitsRecognize)
endif()
