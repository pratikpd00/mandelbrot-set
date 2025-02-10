#include "interactableImage.h"

void InteractableImage::update() {
	for (auto i = 0; i < this->grid->getSizeX(); i++) {
		for (auto j = 0; j < this->grid->getSizeY(); j++) {
			this->image.setPixelColor(i, j, QColor(this->grid->get(i, j)));
		}
	}

	emit newPixmap(QPixmap::fromImage(image));
}

InteractableImage::InteractableImage(unique_ptr<ImageTransformGrid> grid) : grid(std::move(grid)) {
	image = QImage(this->grid->getSizeX(), this->grid->getSizeY(), QImage::Format_ARGB32);
	this->grid->setColoring(ColoringFunctionType::BLUE);
}

QPixmap InteractableImage::getPixmap() const {
	return QPixmap::fromImage(image);
}


//Slots
void InteractableImage::pan(QPoint delta) {
	this->grid->translate(delta.x(), delta.y());
	this->update();
}

void InteractableImage::zoom(double scale, QPoint center) {
	this->grid->zoom(scale, center.x(), center.y());
	this->update();
}