#ifndef MANDELBROTVIEWER_H
#define MANDELBROTVIEWER_H

#include <QMainWindow>
#include <QGraphicsScene>

#include <memory>

#include "interactableImage.h"

namespace Ui {
class MandelbrotViewer;
}

class MandelbrotViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit MandelbrotViewer(QWidget *parent = nullptr);
    ~MandelbrotViewer();

public slots:
	void update(const QPixmap& pixmap);

private:
    Ui::MandelbrotViewer *ui;
    unique_ptr<InteractableImage> image;
    unique_ptr<QGraphicsScene> scene;
    QGraphicsPixmapItem* scenePixmap;
};

#endif // MANDELBROTVIEWER_H
