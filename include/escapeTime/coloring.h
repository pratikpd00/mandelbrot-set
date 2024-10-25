#ifndef COLORING_H
#define COLORING_H

#include <cstdint>

#define ALPHA_OPAQUE 0xFF000000

typedef uint32_t RGBColor;

typedef RGBColor (*ColoringFunction) (int);

inline RGBColor color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = ALPHA_OPAQUE) {
	return (a << 24) | (r << 16) | (g << 8) | b;
}

#endif