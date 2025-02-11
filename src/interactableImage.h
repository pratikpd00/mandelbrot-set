#ifndef INTERACTABLE_IMAGE_H
#define INTERACTABLE_IMAGE_H


#include <memory>

#include <QObject>
#include <QImage>
#include <QPixmap>

#include "escapeTime/ImageTransformGrid.h"

using namespace std;

//Class to handle animations of the mandelbrot set. Since this object inherits from QObject, it can be used with Qt signals and slots, unlike the
//wrapper classes.
class InteractableImage : public QObject {
	Q_OBJECT

	unique_ptr<ImageTransformGrid> grid;
	QImage image;


public:
	InteractableImage(ImageTransformGrid* grid);
	void update();
	QPixmap getPixmap() const;


public slots:
	void pan(QPoint delta);
	void zoom(double factor, QPoint center);

signals:
	void newPixmap(QPixmap map);
};


#endif // !INTERACTABLE_IMAGE_H