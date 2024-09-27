#ifndef MANDELBROTVIEWER_H
#define MANDELBROTVIEWER_H

#include <memory>

#include <QDialog>
#include <QLabel>
#include <QThread>

#include "interactableImage.h"

namespace Ui {
class mandelbrotViewer;
}

class mandelbrotViewer : public QDialog
{
    Q_OBJECT

signals:
    void resize(QSize size);

public:
    explicit mandelbrotViewer(QWidget *parent = nullptr);
    ~mandelbrotViewer();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    Ui::mandelbrotViewer *ui;
};


class InteractiveImageDisplay : public QLabel {
    
    Q_OBJECT

    std::unique_ptr<InteractableImage> image;
    QThread processingThread;
public:
    InteractiveImageDisplay(QWidget *parent = nullptr);
    InteractiveImageDisplay(InteractableImage* image);
    void setImage(InteractableImage* image);

public slots:
    void newSize(QSize size);
};

#endif // MANDELBROTVIEWER_H
