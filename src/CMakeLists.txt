set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

qt_add_executable(MandelbrotViewer
	interactableImage.cpp
    mandelbrotViewer.h 
	mandelbrotViewer.cpp 
	mandelbrotViewer.ui

)

target_link_libraries(MandelbrotViewer PRIVATE
	Qt::Core
    Qt::Gui
	Qt::Widgets
	escapeTime
)

target_include_directories(MandelbrotViewer PUBLIC ${CMAKE_SOURCE_DIR}/include)
