#ifndef MANDELBROTVIEWER_H
#define MANDELBROTVIEWER_H

#include <QMainWindow>
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
	void update(QPixmap pixmap);

private:
    Ui::MandelbrotViewer *ui;
    unique_ptr<InteractableImage> image;
};

#endif // MANDELBROTVIEWER_H
