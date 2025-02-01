#include "mandelbrotViewer.h"

#include <QApplication>
#include <QGraphicsScene>

#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "ui_mandelbrotViewer.h"

MandelbrotViewer::MandelbrotViewer(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MandelbrotViewer)
{
    ui->setupUi(this);
    unique_ptr<ImageTransformGrid> transformGrid(new CudaMandelbrotImageTransformGrid(size().width(), size().height(), 200, 0.1, -1, -1));
    image = unique_ptr<InteractableImage>(new InteractableImage(transformGrid));
    connect(image.get(), &InteractableImage::newPixmap, this, &MandelbrotViewer::update);
    image->update();
}

void MandelbrotViewer::update(QPixmap pixmap) {
    auto scene = new QGraphicsScene(this);
    scene->addPixmap(pixmap);
    scene->setSceneRect(pixmap.rect());
    ui->graphicsView->setScene(scene);

}

MandelbrotViewer::~MandelbrotViewer()
{
    delete ui;
}


int main(int argc, char** argv) {
    QApplication app(argc, argv);

    auto viewer = MandelbrotViewer();
    viewer.show();

    return app.exec();
}