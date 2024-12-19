#ifndef COLORING_H
#define COLORING_H

#include <cstdint>

#include "util.h"

#define ALPHA_OPAQUE 0xFF000000

typedef uint32_t RGBColor;

inline RGBColor color(uint8_t r, uint8_t g, uint8_t b) {
	return ALPHA_OPAQUE | (r << 16) | (g << 8) | b;
}

/* Cuda does not support passing function pointers to kernels, so we use an enum instead */
enum class ColoringFunctionType {
	DEFAULT,
};

#endif