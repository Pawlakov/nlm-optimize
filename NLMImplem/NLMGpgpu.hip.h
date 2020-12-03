#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "NLM.h"

#define HIP_CHECK(command) \
    { \
        hipError_t status = command; \
        if (status!=hipSuccess) \
        { \
            std::cerr << "Error HIP reports " << hipGetErrorString(status) << std::endl; \
            std::abort(); \
        } \
    }

#define MAX(i,j) ( (i)<(j) ? (j):(i) )
#define MIN(i,j) ( (i)<(j) ? (i):(j) )

#define dTiny 1e-10
#define fTiny 0.00000001f
#define fLarge 100000000.0f

///// LUT tables
#define LUTMAX 30.0
#define LUTMAXM1 29.0
#define LUTPRECISION 1000.0

void  wxFillExpLut(float* lut, int size);        // Fill exp(-x) lut

float wxSLUT(float dif, float* lut);                     // look at LUT

void fpClear(float* fpI, float fValue, int iLength);

float fiL2FloatDist(float* u0, float* u1, int i0, int j0, int i1, int j1, int radius, int width0, int width1);

float fiL2FloatDist(float** u0, float** u1, int i0, int j0, int i1, int j1, int radius, int channels, int width0, int width1);