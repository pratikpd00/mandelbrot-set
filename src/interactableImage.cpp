#include "interactableImage.h"

void InteractableImage::update() {
	for (auto i = 0; i < this->grid->getSizeX(); i++) {
		for (auto j = 0; j < this->grid->getSizeY(); j++) {
			this->image.setPixelColor(i, j, QColor(this->grid->get(i, j)));
		}
	}

	emit newPixmap(QPixmap::fromImage(image));
}

InteractableImage::InteractableImage(ImageTransformGrid* grid)  {
	this->grid = unique_ptr<ImageTransformGrid>(grid);
	image = QImage(this->grid->getSizeX(), this->grid->getSizeY(), QImage::Format_ARGB32);
	grid->resizeGrid(image.width() + 1, image.height() + 1);
	grid->resizeGrid(image.width(), image.height());
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