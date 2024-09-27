#include "interactableImage.h"


InteractableImage* InteractableImage::resized(QSize size) {
	return this->resized(size.width(), size.height());
}


MandelbrotEscapeTimeImage::MandelbrotEscapeTimeImage(int sizeX, int sizeY, escapeTimeAlgorithm escapeTime) : InteractableImage(sizeX, sizeY, QImage::Format_RGB32)
{
	this->escapeTime = escapeTime;
	mandelbrotEscapeTimeGrid = std::vector<int>(sizeX * sizeY);
	this->iters = 100;
	this->scale = 0.005;
	this->yStart = this->xStart = -1.5;
	escapeTime(mandelbrotEscapeTimeGrid.data(), iters, sizeX, sizeY, scale, xStart, yStart);
	update();
}

void MandelbrotEscapeTimeImage::pan(int xPan, int yPan) {
	this->yStart += yPan * this->scale;
	this->xStart += xPan * this->scale;

	escapeTime(this->mandelbrotEscapeTimeGrid.data(), this->iters, this->size().width(), this->size().height(), this->scale, this->xStart, this->yStart);

	update();
}

InteractableImage* MandelbrotEscapeTimeImage::resized(int width, int height) {
	return new MandelbrotEscapeTimeImage(width, height, this->escapeTime);
}

void MandelbrotEscapeTimeImage::update()
{
	for (int i = 0; i < this->size().height(); i++) {
		for (int j = 0; j < this->size().width(); j++) {
			QRgb color;
			auto escapeTime = mandelbrotEscapeTimeGrid[j * size().height() + i];

			if (escapeTime == this->iters) {
				color = QColor("black").rgb();
			}
			else {
				auto factor = escapeTime / (double)this->iters;
				color = qRgb(0, 0, 255 * factor);
			}

			this->setPixelColor(j, i, color);
		}
	}

}