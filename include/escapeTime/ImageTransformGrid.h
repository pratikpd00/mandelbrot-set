#ifndef IMAGETRANSFORMGRID_H
#define IMAGETRANSFORMGRID_H

#include <cstdint>
#define ALPHA_OPAQUE 0xFF000000

typedef uint32_t RGBColor;

class ImageTransformGrid {
public:
	virtual ~ImageTransformGrid() {};
	virtual void zoom(double scale) = 0;
	virtual void resize(int sizeX, int sizeY, int centerX, int centerY) = 0;
	virtual void translate(double panX, double panY) = 0;
	virtual RGBColor get(int x, int, y) = 0;
};

inline RGBColor color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = ALPHA_OPAQUE) {
	return (a << 24) | (r << 16) | (g << 8) | b;
}

#endif