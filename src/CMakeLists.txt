set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

qt_add_executable(MandelbrotViewer
	mandelbrotViewer.cpp
	mandelbrotViewer.ui
)

target_link_libraries(MandelbrotViewer PUBLIC
	Qt::Core
    Qt::Gui
	Qt::Widgets
	escapeTime
)