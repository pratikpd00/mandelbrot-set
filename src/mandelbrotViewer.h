#ifndef MANDELBROTVIEWER_H
#define MANDELBROTVIEWER_H

#include <QDialog>

namespace Ui {
class mandelbrotViewer;
}

class mandelbrotViewer : public QDialog
{
    Q_OBJECT

signals:
    void loadPixmap(QPixmap);

public:
    explicit mandelbrotViewer(QWidget *parent = nullptr);
    ~mandelbrotViewer();

private:
    Ui::mandelbrotViewer *ui;
};

#endif // MANDELBROTVIEWER_H
