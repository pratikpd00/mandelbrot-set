#ifndef COLORING_H
#define COLORING_H

#include <cstdint>

#define ALPHA_OPAQUE 0xFF000000

typedef uint32_t RGBColor;

typedef RGBColor (*ColoringFunction) (int escapeTime, int maxEscapeTime);

inline RGBColor color(uint8_t r, uint8_t g, uint8_t b) {
	return ALPHA_OPAQUE | (r << 16) | (g << 8) | b;
}

#endif