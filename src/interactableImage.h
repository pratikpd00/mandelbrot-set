#ifndef INTERACTABLE_IMAGE_H
#define INTERACTABLE_IMAGE_H


#include <QImage>
#include <vector>

#include "escapeTime/types.h"

#include <memory>

#include <QObject>
#include <QImage>

#include "escapeTime/ImageTransformGrid.h"

using namespace std;

class InteractableImage : public QObject {
	Q_OBJECT

	unique_ptr<ImageTransformGrid> grid;
	QImage image;

public:
	InteractableImage(unique_ptr<ImageTransformGrid> grid);

public slots:
	void update();

signals:
	void newPixmap(QPixmap map);
};


#endif // !INTERACTABLE_IMAGE_H