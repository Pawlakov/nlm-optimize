#pragma once

extern "C" __declspec(dllexport) void Denoise(int iDWin, int iDBloc, float fSigma, float fFiltPar, float* fpI, float* fpO, int iChannels, int iWidth, int iHeight);