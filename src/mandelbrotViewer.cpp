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
    unique_ptr<ImageTransformGrid> transformGrid(new CudaMandelbrotImageTransformGrid(size.width(), size.height(), 200, 0.005, -2, -1.5));
    image = unique_ptr<InteractableImage>(new InteractableImage(std::move(transformGrid)));
    scene = make_unique<QGraphicsScene>(this);
    image->update();
    scenePixmap = scene->addPixmap(image->getPixmap());
    ui->mandelbrotView->setScene(scene.get());
    connect(image.get(), &InteractableImage::newPixmap, this, &MandelbrotViewer::update);
}

void MandelbrotViewer::update(const QPixmap& pixmap) {
    scenePixmap->setPixmap(pixmap);
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