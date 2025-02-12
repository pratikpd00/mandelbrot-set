#ifndef COLORING_H
#define COLORING_H

#include <cstdint>

#include "util.h"

#define ALPHA_OPAQUE 0xFF000000

typedef uint32_t RGBColor;

/* Cuda does not support passing function pointers to kernels, so we use an enum instead */
enum class ColoringFunctionType {
	DEFAULT,
	BLUE
};

#endif