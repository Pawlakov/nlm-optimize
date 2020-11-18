#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

extern void Denoise(int iDWin, int iDBloc, float fSigma, float fFiltPar, float** fpI, float** fpO, int iChannels, int iWidth, int iHeight);