#include "mandelbrotViewer.h"

#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>

#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "ui_mandelbrotViewer.h"

MandelbrotViewer::MandelbrotViewer(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MandelbrotViewer)
{
    ui->setupUi(this);
    ui->mandelbrotView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->mandelbrotView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->mandelbrotView->resize(this->size());

    auto size = ui->mandelbrotView->size();

    ImageTransformGrid* transformGrid = new CudaMandelbrotImageTransformGrid(size.width(), size.height(), 200, 0.005, -2, -1.5);
    image = unique_ptr<InteractableImage>(new InteractableImage(transformGrid));

    image->moveToThread(&processingThread);
    processingThread.start();
    image->update();

    ui->mandelbrotView->setPixmap(image->getPixmap());
    connect(image.get(), &InteractableImage::newPixmap, ui->mandelbrotView, &InteractableImageView::update);
    connect(ui->mandelbrotView, &InteractableImageView::pan, image.get(), &InteractableImage::pan);
    connect(ui->mandelbrotView, &InteractableImageView::zoom, image.get(), &InteractableImage::zoom);

    connect(ui->zoomInButton, &QAbstractButton::clicked, image.get(), &InteractableImage::zoomIn);
    connect(ui->zoomOutButton, &QAbstractButton::clicked, image.get(), &InteractableImage::zoomOut);
}

MandelbrotViewer::~MandelbrotViewer()
{
    processingThread.quit();
    processingThread.wait();
    delete ui;
}


int main(int argc, char** argv) {
    QApplication app(argc, argv);

    auto viewer = MandelbrotViewer();
    viewer.show();

    return app.exec();
}