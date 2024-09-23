#ifndef INTERACTABLE_IMAGE_H
#define INTERACTABLE_IMAGE_H


#include <QImage>
#include <vector>

#include "escapeTime/types.h"

/*
* Intended as a common interface for any pannable and zoomable image.
* Should not be implemented directly, and instead should be implemented by child classes.
*/
class InteractableImage : public QImage {
public:
	InteractableImage(int sizeX, int sizeY, QImage::Format format) : QImage(sizeX, sizeY, format) {};
	//virtual void zoom(double factor) = 0;
	//virtual void pan(int xPan, int yPan) = 0;
	//virtual void update() = 0;
	//virtual int scaled(int width, int height, Qt::AspectRatioMode aspectRatioMode = Qt::IgnoreAspectRatio, Qt::TransformationMode transformMode = Qt::FastTransformation) const = 0;
};

class MandelbrotEscapeTimeImage : public InteractableImage {

	//Q_OBJECT

	escapeTimeAlgorithm escapeTime;
	std::vector<int> mandelbrotEscapeTimeGrid;
	int iters;
	double scale;
	double xStart;
	double yStart;

	void update();
public:
	MandelbrotEscapeTimeImage(int sizeX, int sizeY, escapeTimeAlgorithm escapeTime);
	//void zoom(double factor);
	//void pan(int xPan, int yPan);
	//int scaled(int width, int height, Qt::AspectRatioMode aspectRatioMode = Qt::IgnoreAspectRatio, Qt::TransformationMode transformMode = Qt::FastTransformation) const;
};

#endif // !INTERACTABLE_IMAGE_H