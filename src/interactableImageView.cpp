#include "interactableImageView.h"

#include <QGraphicsPixmapItem>
#include <QMouseEvent>

#define WHEEL_ZOOM_FACTOR 0.02

InteractableImageView::InteractableImageView(QWidget *parent)
	: QGraphicsView(parent)
{
	scene = make_unique<QGraphicsScene>(this);
	scenePixmap = scene->addPixmap(QPixmap());
	setScene(scene.get());

	backgroundBrush = QBrush(Qt::black);
	setBackgroundBrush(backgroundBrush);
}

void InteractableImageView::setPixmap(const QPixmap & pixmap) {

	scenePixmap->setPixmap(pixmap);
}

void InteractableImageView::mousePressEvent(QMouseEvent* event) {
	if (event->button() == Qt::LeftButton) {
		mousePressed = true;
		prevMousePosition = event->pos();
		initialMousePosition = event->pos();
		event->accept();
	}
	event->ignore();
}

void InteractableImageView::mouseMoveEvent(QMouseEvent* event) {
	if (mousePressed) {
		auto d = event->pos() - prevMousePosition;
		prevMousePosition = event->pos();
		scrollContentsBy(d.x(), d.y());
		event->accept();
	}
	event->ignore();
}

void InteractableImageView::mouseReleaseEvent(QMouseEvent* event) {
	if (event->button() == Qt::LeftButton) {
		mousePressed = false;
		emit pan(event->pos() - initialMousePosition);
		event->accept();
	}
	event->ignore();
}

void InteractableImageView::update(const QPixmap pixmap) {
	scenePixmap->setPixmap(pixmap);
}


void InteractableImageView::wheelEvent(QWheelEvent* event) {
	double scale;
	if (event->angleDelta().y() > 0) {
		scale = 1 / (event->angleDelta().y() * WHEEL_ZOOM_FACTOR);
	} else {
		scale = -1 * event->angleDelta().y() * WHEEL_ZOOM_FACTOR;
	}
	emit zoom(scale, event->position().toPoint());
	event->accept();
}

InteractableImageView::~InteractableImageView() {
}
