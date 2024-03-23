#ifndef ANACAL_CONVOLVE_H
#define ANACAL_CONVOLVE_H

#include <fftw3.h>
#include <image.h>

namespace anacal {
    void convolve(Image& img, const BaseModel& filter_mod);
    void convolve(Image& img, const Image& filter_img);

    void deconvolve(Image& img, const BaseModel& filter_mod);
    void deconvolve(Image& img, const Image& filter_img);
}

#endif
