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