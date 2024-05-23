#ifndef TEST_HISTOGRAM_CUH
#define TEST_HISTOGRAM_CUH

#include "wb.h"


namespace cp {
    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations = 1);
}



#endif //TEST_HISTOGRAM_CUH
