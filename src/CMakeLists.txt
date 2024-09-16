add_subdirectory(mandelbrotDisplay)

qt_add_executable(MandelbrotViewer
	interactableImage.cpp
	mandelbrotViewer.cpp
	mandelbrotViewer.ui
)

target_link_libraries(MandelbrotViewer PRIVATE Qt6::Widgets)