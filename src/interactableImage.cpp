#include "interactableImage.h"

void InteractableImage::update() {
	for (auto i = 0; i < this->grid->getSizeX(); i++) {
		for (auto j = 0; j < this->grid->getSizeY(); j++) {
			this->image.setPixel(i, j, this->grid->get(i, j));
		}
	}

	emit newPixmap(QPixmap::fromImage(image));
}

InteractableImage::InteractableImage(unique_ptr<ImageTransformGrid> grid) : grid(std::move(grid)) {
	image = QImage(this->grid->getSizeX(), this->grid->getSizeY(), QImage::Format_ARGB32);
}

