#include <qapplication>

#include "mandelbrotViewer.h"
#include "ui_mandelbrotViewer.h"
#include "interactableImage.h"
#include "escapeTime/escapeTimeCuda.h"

mandelbrotViewer::mandelbrotViewer(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::mandelbrotViewer)
{
    ui->setupUi(this);
    auto image = MandelbrotEscapeTimeImage(811, 521, escapeTimeCUDA);
    QPixmap map(811, 521);
    map.convertFromImage(image);
    connect(this, &mandelbrotViewer::loadPixmap, ui->mandelbrotImage, &QLabel::setPixmap);
    emit loadPixmap(map);
}

mandelbrotViewer::~mandelbrotViewer()
{
    delete ui;
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    mandelbrotViewer viewer;
    viewer.show();
    return app.exec();
}