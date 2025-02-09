#pragma once

#include <QGraphicsView>
#include <memory>

using namespace std;

//Since this program needs to be able to handle mouse events, this class need to inherit from QGraphicsView to override those functions.
class InteractableImageView  : public QGraphicsView
{
	Q_OBJECT

	QGraphicsPixmapItem *scenePixmap;
	unique_ptr<QGraphicsScene> scene;

	bool mousePressed = false;
	QPoint prevMousePosition = QPoint(0, 0);
	QPoint initialMousePosition = QPoint(0, 0);
	QBrush backgroundBrush;

public:
	InteractableImageView(QWidget *parent);
	void setPixmap(const QPixmap& pixmap);
	void mousePressEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent *event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;
	~InteractableImageView();

public slots:
	void update(const QPixmap& pixmap);

signals:
	void pan(QPoint point);
};
