#include "interactableImage.h"


MandelbrotEscapeTimeImage::MandelbrotEscapeTimeImage(int sizeX, int sizeY, escapeTimeAlgorithm escapeTime) : InteractableImage(sizeX, sizeY, QImage::Format_RGB32)
{
	this->escapeTime = escapeTime;
	mandelbrotEscapeTimeGrid = std::vector<int>(sizeX * sizeY);
	this->iters = 100;
	this->scale = 0.01;
	this->yStart = this->xStart = -1;
	escapeTime(mandelbrotEscapeTimeGrid.data(), iters, sizeX, sizeY, scale, xStart, yStart);
	update();
}

void MandelbrotEscapeTimeImage::update()
{
	for (int i = 0; i < this->size().height(); i++) {
		for (int j = 0; j < this->size().width(); j++) {
			QRgb color;
			auto escapeTime = mandelbrotEscapeTimeGrid[i * size().height() + j];

			if (escapeTime == this->iters) {
				color = QColor("black").rgb();
			}
			else {
				auto factor = escapeTime / (double)this->iters;
				color = qRgb(0, 0, 255 * factor);
			}

			this->setPixelColor(i, j, color);
		}
	}

}
