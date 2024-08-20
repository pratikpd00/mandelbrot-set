#include <gtest/gtest.h>
#include "escapeTime/escapeTimeCuda.h"

TEST(TestCudaEscapeTime, OutOfRadius) {
	int escapeTime;
	int* escapeTimePtr = &escapeTime;
	escapeTimeCUDA(escapeTimePtr, 1000, 1, 1, 1.0, 4.0, 4.0);
	ASSERT_EQ(escapeTime, 0);

}