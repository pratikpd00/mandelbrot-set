#ifndef INTERACTABLE_IMAGE_H
#define INTERACTABLE_IMAGE_H


#include <memory>

#include <QObject>
#include <QImage>
#include <QPixmap>

#include "escapeTime/ImageTransformGrid.h"

using namespace std;

class InteractableImage : public QObject {
	Q_OBJECT

	unique_ptr<ImageTransformGrid> grid;
	QImage image;


public:
	InteractableImage(unique_ptr<ImageTransformGrid> grid);
	void update();
	QPixmap getPixmap() const;


public slots:
	

signals:
	void newPixmap(QPixmap map);
};


#endif // !INTERACTABLE_IMAGE_H