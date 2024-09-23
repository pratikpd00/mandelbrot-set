#ifndef MANDELBROTVIEWER_H
#define MANDELBROTVIEWER_H

#include <QDialog>
#include <QLabel>

#include "interactableImage.h"

namespace Ui {
class mandelbrotViewer;
}

class mandelbrotViewer : public QDialog
{
    Q_OBJECT

signals:
    void loadPixmap(QImage);

public:
    explicit mandelbrotViewer(QWidget *parent = nullptr);
    ~mandelbrotViewer();

private:
    Ui::mandelbrotViewer *ui;
};


class InteractiveImageDisplay : public QLabel {
    
    Q_OBJECT

    InteractableImage* image;
public:
    InteractiveImageDisplay(QWidget *parent = nullptr);
    InteractiveImageDisplay(InteractableImage* image);
    void setImage(InteractableImage* image);
};

#endif // MANDELBROTVIEWER_H
