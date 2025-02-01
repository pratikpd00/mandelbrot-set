#include <qapplication>

#include "mandelbrotViewer.h"
#include "ui_mandelbrotViewer.h"
#include "interactableImage.h"

mandelbrotViewer::mandelbrotViewer(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::mandelbrotViewer) {
	ui->setupUi(this);
}

mandelbrotViewer::~mandelbrotViewer() {
    delete ui;
}

void mandelbrotViewer::resizeEvent(QResizeEvent* event) {
    this->QDialog::resizeEvent(event);
    emit this->resize(this->size());
}

InteractiveImageDisplay::InteractiveImageDisplay(QWidget* parent) : QLabel(parent) {
    if (parent != nullptr) {
        this->resize(parent->size());
    }
}

InteractiveImageDisplay::InteractiveImageDisplay(InteractableImage* image) : QLabel() {
    this->setImage(image);
}

void InteractiveImageDisplay::setImage(InteractableImage* image) {
    this->image = std::unique_ptr<InteractableImage>(image->resized(this->size()));
    this->setPixmap(QPixmap::fromImage(*(this->image)));
}

void InteractiveImageDisplay::newSize(QSize size) {
    this->resize(size);
    this->setImage(this->image->resized(size));
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    mandelbrotViewer viewer;
    viewer.show();
    return app.exec();
}


