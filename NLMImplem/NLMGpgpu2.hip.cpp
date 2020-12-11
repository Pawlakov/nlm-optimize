#include "NLMGpgpu2.hip.h"

void fpClear(float* fpI, float fValue, int iLength)
{
	for (int ii = 0; ii < iLength; ii++)
	{
		fpI[ii] = fValue;
	}
}

// LUT tables
void  wxFillExpLut(float* lut, int size)
{
	for (int i = 0; i < size; i++)
	{
		lut[i] = expf(-(float)i / LUTPRECISION);
	}
}

__device__ float wxSLUT(float dif, float* lut)
{
	if (dif >= (float)LUTMAXM1)
	{
		return 0.0;
	}

	int  x = (int)floor((double)dif * (float)LUTPRECISION);

	float y1 = lut[x];
	float y2 = lut[x + 1];

	return y1 + (y2 - y1) * (dif * LUTPRECISION - x);
}

__device__ float fiL2FloatDist(float* u0, float* u1, int i0, int j0, int i1, int j1, int radius, int width0, int width1)
{
	float dist = 0.0;
	for (int s = -radius; s <= radius; s++)
	{
		int l = (j0 + s) * width0 + (i0 - radius);
        int ptr0 = l;

        l = (j1 + s) * width1 + (i1 - radius);
        int ptr1 = l;

        for (int r = -radius; r <= radius; r++, ptr0++, ptr1++)
        {
            float dif = u0[ptr0] - u1[ptr1];
            dist += (dif * dif);
        }
	}

	return dist;
}

__device__ float fiL2FloatDist(float* u0, float* u1, int i0, int j0, int i1, int j1, int radius, int channels, int width0, int width1, int iwxh)
{
	float dif = 0.0f;
	for (int ii = 0; ii < channels; ii++)
	{
		dif += fiL2FloatDist(&u0[ii * iwxh], &u1[ii * iwxh], i0, j0, i1, j1, radius, width0, width1);
	}

	return dif;
}

__global__ void shitKernel(int iDWin, int iDBloc, float fSigma, float fFiltPar, float* fpI, int iWidth, int iHeight, int iChannels, int x, int y, int imin, int imax, int jmin, int jmax, float* fpWeights, float* fpODenoised, float* fpLut)
{
    int iwxh = iWidth * iHeight;
    int ihwl = (2 * iDWin + 1);
    int iwl = (2 * iDWin + 1) * (2 * iDWin + 1);
    int icwl = iChannels * iwl;
    float fSigma2 = fSigma * fSigma;
	float fH = fFiltPar * fSigma;
	float fH2 = fH * fH;
    int iDWin0 = MIN(iDWin, MIN(iWidth - 1 - x, MIN(iHeight - 1 - y, MIN(x, y))));

    int iKernelId = threadIdx.x + blockIdx.x * blockDim.x;
    int iLength = imax - imin + 1;
    int j = jmin + (iKernelId / iLength);
    int i = imin + (iKernelId % iLength);

    if (j <= jmax && i <= imax)
    {
        if (i != x || j != y)
        {
            float fDif = fiL2FloatDist(fpI, fpI, x, y, i, j, iDWin0, iChannels, iWidth, iWidth, iwxh);

            // dif^2 - 2 * fSigma^2 * N      dif is not normalized
            fDif = MAX(fDif - 2.0f * (float)icwl * fSigma2, 0.0f);
            fDif = fDif / fH2;

            float fWeight = wxSLUT(fDif, fpLut);

            fpWeights[iKernelId] = fWeight;

            for (int is = -iDWin0; is <= iDWin0; is++)
            {
                int aiindex = (iDWin + is) * ihwl + iDWin;
                int ail = (j + is) * iWidth + i;

                for (int ir = -iDWin0; ir <= iDWin0; ir++)
                {
                    int iindex = aiindex + ir;
                    int il = ail + ir;

                    for (int ii = 0; ii < iChannels; ii++)
                    {
                        fpODenoised[ii * iwl + iindex] += fWeight * fpI[ii * iwxh + il];
                    }
                }
            }
        }
    }
}

extern void Denoise(int iDWin, int iDBloc, float fSigma, float fFiltPar, float** fpI, float** fpO, int iChannels, int iWidth, int iHeight)
{
	// length of each channel
	int iwxh = iWidth * iHeight;

	//  length of comparison window
	int ihwl = (2 * iDWin + 1);
	int iwl = (2 * iDWin + 1) * (2 * iDWin + 1);
	int icwl = iChannels * iwl;

	// filtering parameter
	//float fSigma2 = fSigma * fSigma;
	float fH = fFiltPar * fSigma;
	float fH2 = fH * fH;

	// multiply by size of patch, since distances are not normalized
	fH2 *= (float)icwl;

	// tabulate exp(-x), faster than using directly function expf
	int iLutLength = (int)rintf((float)LUTMAX * (float)LUTPRECISION);
	float* fpLut = new float[iLutLength];
	wxFillExpLut(fpLut, iLutLength);

	// auxiliary variable
	// number of denoised values per pixel
	float* fpCount = new float[iwxh];
	fpClear(fpCount, 0.0f, iwxh);

	// clear output
	for (int ii = 0; ii < iChannels; ii++)
	{
		fpClear(fpO[ii], 0.0f, iwxh);
	}

    for (int y = 0; y < iHeight; y++)
    {
        // auxiliary variable
        // denoised patch centered at a certain pixel
        float** fpODenoised = new float* [iChannels];
        for (int ii = 0; ii < iChannels; ii++)
        {
            fpODenoised[ii] = new float[iwl];
        }

        for (int x = 0; x < iWidth; x++)
        {
            // reduce the size of the comparison window if we are near the boundary
            int iDWin0 = MIN(iDWin, MIN(iWidth - 1 - x, MIN(iHeight - 1 - y, MIN(x, y))));

            // research zone depending on the boundary and the size of the window
            int imin = MAX(x - iDBloc, iDWin0);
            int jmin = MAX(y - iDBloc, iDWin0);

            int imax = MIN(x + iDBloc, iWidth - 1 - iDWin0);
            int jmax = MIN(y + iDBloc, iHeight - 1 - iDWin0);

            //  clear current denoised patch
            for (int ii = 0; ii < iChannels; ii++) 
            {
                fpClear(fpODenoised[ii], 0.0f, iwl);
            }

            // maximum of weights. Used for reference patch
            float fMaxWeight = 0.0f;

            // sum of weights
            float fTotalWeight = 0.0f;

            int iKernelCount = (jmax - jmin + 1) * (imax - imin + 1);
            dim3 gpuThreads(256, 1, 1);
            dim3 gpuBlocks((iKernelCount + 256 - 1) / 256, 1, 1);

            float* fpIDevice;
            float* fpWeightsDevice;
            float* fpODenoisedDevice;
            float* fpLutDevice;
            HIP_CHECK(hipMalloc(&fpIDevice, sizeof(float) * iwxh * iChannels));
            HIP_CHECK(hipMalloc(&fpWeightsDevice, sizeof(float) * iKernelCount));
            HIP_CHECK(hipMalloc(&fpODenoisedDevice, sizeof(float) * iwl * iChannels));
            HIP_CHECK(hipMalloc(&fpLutDevice, sizeof(float) * iLutLength));
            // zrobić tu memcpy

            hipLaunchKernelGGL(shitKernel, gpuBlocks, gpuThreads, 0, 0, iDWin, iDBloc, fSigma, fFiltPar, fpIDevice, iWidth, iHeight, iChannels, x, y, imin, imax, jmin, jmax, fpWeightsDevice, fpODenoisedDevice, fpLutDevice);
            HIP_CHECK(hipGetLastError());

            // HERE SHALL THE KERNEL BE CALLED
            /*
                if (fWeight > fMaxWeight)
                {
                    fMaxWeight = fWeight;
                }

                fTotalWeight += fWeight;
            */

            // current patch with fMaxWeight
            for (int is = -iDWin0; is <= iDWin0; is++)
            {
                int aiindex = (iDWin + is) * ihwl + iDWin;
                int ail = (y + is) * iWidth + x;
                for (int ir = -iDWin0; ir <= iDWin0; ir++)
                {
                    int iindex = aiindex + ir;
                    int il = ail + ir;

                    for (int ii = 0; ii < iChannels; ii++)
                    {
                        fpODenoised[ii][iindex] += fMaxWeight * fpI[ii][il];
                    }
                }
            }

            fTotalWeight += fMaxWeight;

            // normalize average value when fTotalweight is not near zero
            if (fTotalWeight > fTiny)
            {
                for (int is = -iDWin0; is <= iDWin0; is++)
                {
                    int aiindex = (iDWin + is) * ihwl + iDWin;
                    int ail = (y + is) * iWidth + x;

                    for (int ir = -iDWin0; ir <= iDWin0; ir++)
                    {
                        int iindex = aiindex + ir;
                        int il = ail + ir;

                        fpCount[il]++;

                        for (int ii = 0; ii < iChannels; ii++)
                        {
                            fpO[ii][il] += fpODenoised[ii][iindex] / fTotalWeight;
                        }
                    }
                }
            }
        }

        for (int ii = 0; ii < iChannels; ii++)
        {
            delete[] fpODenoised[ii];
        }

        delete[] fpODenoised;
    }

	for (int ii = 0; ii < iwxh; ii++)
	{
		if (fpCount[ii] > 0.0)
		{
			for (int jj = 0; jj < iChannels; jj++)
			{
				fpO[jj][ii] /= fpCount[ii];
			}
		}
		else
		{
			for (int jj = 0; jj < iChannels; jj++)
			{
				fpO[jj][ii] = fpI[jj][ii];
			}
		}
	}

	// delete memory
	delete[] fpLut;
	delete[] fpCount;
}