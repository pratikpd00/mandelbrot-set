#include "interactableImage.h"

InteractableImage::InteractableImage(unique_ptr<ImageTransformGrid> grid) : grid(std::move(grid)) {
	image = QImage(this->grid->getSizeX(), this->grid->getSizeY(), QImage::Format_ARGB32);
}