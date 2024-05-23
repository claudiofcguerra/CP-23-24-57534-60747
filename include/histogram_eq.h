#ifndef CP_PROJECT_HISTOGRAM_EQ_H
#define CP_PROJECT_HISTOGRAM_EQ_H

#include "wb.h"

namespace cp
{
    enum class func_type { ORG, OMP, CUDA };

    wbImage_t iterative_histogram_equalization(const wbImage_t& input_image, int iterations = 1,
                                               func_type type = func_type::ORG);
}

#endif //CP_PROJECT_HISTOGRAM_EQ_H
