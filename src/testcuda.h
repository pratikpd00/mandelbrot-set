#include "escapeTime/mandelbrotImageTransformGrid.h"
#include <QObject>



class cuddatest : public QObject {

	Q_OBJECT
	ImageTransformGrid* grid = new CudaMandelbrotImageTransformGrid(2000, 1000, 200, 0.005, -2, -1.5);
public:


	void zoom() { grid->zoom(2, 0, 0); }
	cuddatest() : QObject() {grid->setColoring(Coloring::BLUE); }
	~cuddatest() { delete grid; }
};