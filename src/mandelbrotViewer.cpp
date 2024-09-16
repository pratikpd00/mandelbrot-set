#include "mandelbrotViewer.h"
#include "ui_mandelbrotViewer.h"

mandelbrotViewer::mandelbrotViewer(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::mandelbrotViewer)
{
    ui->setupUi(this);
}

mandelbrotViewer::~mandelbrotViewer()
{
    delete ui;
}
