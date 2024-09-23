#include <qapplication>

#include "mandelbrotViewer.h"
#include "ui_mandelbrotViewer.h"
#include "interactableImage.h"
#include "escapeTime/escapeTimeCuda.h"

mandelbrotViewer::mandelbrotViewer(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::mandelbrotViewer) {
    ui->setupUi(this);
    auto image = MandelbrotEscapeTimeImage(811, 521, escapeTimeCUDA);
    ui->mandelbrotImage->setImage(&image);
}

mandelbrotViewer::~mandelbrotViewer() {
    delete ui;
}

InteractiveImageDisplay::InteractiveImageDisplay(QWidget *parent) : QLabel(parent) {}

InteractiveImageDisplay::InteractiveImageDisplay(InteractableImage* image) : QLabel() {
    this->setImage(image);
}

void InteractiveImageDisplay::setImage(InteractableImage* image) {
    this->image = image;
    this->setPixmap(QPixmap::fromImage(*image));
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    mandelbrotViewer viewer;
    viewer.show();
    return app.exec();
}


