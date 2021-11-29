namespace NLMRunner.NLM
{
    using System;
    using System.Threading;
    using System.Threading.Tasks;
    using NLMShared.Models;
    using NLMShared.NLM;

    public unsafe class DefaultImplementation
        : IImplementation
    {
        private const float LUTMAX = 30.0f;
        private const float LUTMAXM1 = 29.0f;
        private const float LUTPRECISION = 1000.0f;

        private const float fTiny = 0.00000001f;
        private const float fLarge = 100000000.0f;

        public unsafe void RunDenoise(float[] inputArray, float[] outputArray, NLMParamsModel nlmParams, int channels, int width, int height)
        {
            fixed (float* inputFlatPointer = &inputArray[0], outputFlatPointer = &outputArray[0])
            {
                var fpI = new float*[channels];
                var fpO = new float*[channels];
                for (int ii = 0; ii < channels; ii++)
                {
                    fpI[ii] = &inputFlatPointer[ii * width * height];
                    fpO[ii] = &outputFlatPointer[ii * width * height];
                }

                fixed (float** inputPointer = &fpI[0], outputPointer = &fpO[0])
                {
                    this.DenoiseBody(nlmParams.Win, nlmParams.Bloc, nlmParams.Sigma, nlmParams.FiltPar, inputPointer, outputPointer, channels, width, height);
                }
            }
        }

        public void Dispose()
        {
        }

        private void DenoiseBody(int iDWin, int iDBloc, float fSigma, float fFiltPar, float** fpI, float** fpO, int iChannels, int iWidth, int iHeight)
        {
            // length of each channel
            var iwxh = iWidth * iHeight;

            // length of comparison window
            var ihwl = ((2 * iDWin) + 1);
            var iwl = ((2 * iDWin) + 1) * ((2 * iDWin) + 1);
            var icwl = iChannels * iwl;

            // filtering parameter
            var fSigma2 = fSigma * fSigma;
            var fH = fFiltPar * fSigma;
            var fH2 = fH * fH;

            // multiply by size of patch, since distances are not normalized
            fH2 *= icwl;

            // tabulate exp(-x), faster than using directly function expf
            var iLutLength = (int)Math.Round(LUTMAX * LUTPRECISION);
            var fpLut = new float[iLutLength];
            this.wxFillExpLut(fpLut, iLutLength);

            // auxiliary variable
            // number of denoised values per pixel
            var fpCount = new float[iwxh];

            Parallel.For(0, iHeight, y =>
            {
                // auxiliary variable
                // denoised patch centered at a certain pixel
                var fpODenoised = new float[iChannels][];
                for (var ii = 0; ii < iChannels; ii++)
                {
                    fpODenoised[ii] = new float[iwl];
                }

                for (var x = 0; x < iWidth; x++)
                {
                    // reduce the size of the comparison window if we are near the boundary
                    var iDWin0 = Math.Min(iDWin, Math.Min(iWidth - 1 - x, Math.Min(iHeight - 1 - y, Math.Min(x, y))));

                    // research zone depending on the boundary and the size of the window
                    var imin = Math.Max(x - iDBloc, iDWin0);
                    var jmin = Math.Max(y - iDBloc, iDWin0);

                    var imax = Math.Min(x + iDBloc, iWidth - 1 - iDWin0);
                    var jmax = Math.Min(y + iDBloc, iHeight - 1 - iDWin0);

                    // clear current denoised patch
                    for (var ii = 0; ii < iChannels; ii++)
                    {
                        for (var iii = 0; iii < iwl; iii++)
                        {
                            fpODenoised[ii][iii] = 0.0f;
                        }
                    }

                    // maximum of weights. Used for reference patch
                    var fMaxWeight = 0.0f;

                    // sum of weights
                    var fTotalWeight = 0.0f;

                    for (var j = jmin; j <= jmax; j++)
                    {
                        for (var i = imin; i <= imax; i++)
                        {
                            if (i != x || j != y)
                            {
                                var fDif = this.fiL2FloatDist(fpI, fpI, x, y, i, j, iDWin0, iChannels, iWidth, iWidth);

                                // dif^2 - 2 * fSigma^2 * N      dif is not normalized
                                fDif = Math.Max(fDif - (2.0f * (float)icwl * fSigma2), 0.0f);
                                fDif = fDif / fH2;

                                var fWeight = this.wxSLUT(fDif, fpLut);

                                if (fWeight > fMaxWeight)
                                {
                                    fMaxWeight = fWeight;
                                }

                                fTotalWeight += fWeight;

                                for (var @is = -iDWin0; @is <= iDWin0; @is++)
                                {
                                    var aiindex = ((iDWin + @is) * ihwl) + iDWin;
                                    var ail = ((j + @is) * iWidth) + i;

                                    for (var ir = -iDWin0; ir <= iDWin0; ir++)
                                    {
                                        var iindex = aiindex + ir;
                                        var il = ail + ir;

                                        for (var ii = 0; ii < iChannels; ii++)
                                        {
                                            fpODenoised[ii][iindex] += fWeight * fpI[ii][il];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // current patch with fMaxWeight
                    for (var @is = -iDWin0; @is <= iDWin0; @is++)
                    {
                        var aiindex = ((iDWin + @is) * ihwl) + iDWin;
                        var ail = ((y + @is) * iWidth) + x;

                        for (var ir = -iDWin0; ir <= iDWin0; ir++)
                        {
                            var iindex = aiindex + ir;
                            var il = ail + ir;

                            for (var ii = 0; ii < iChannels; ii++)
                            {
                                fpODenoised[ii][iindex] += fMaxWeight * fpI[ii][il];
                            }
                        }
                    }

                    fTotalWeight += fMaxWeight;

                    // normalize average value when fTotalweight is not near zero
                    if (fTotalWeight > fTiny)
                    {
                        for (var @is = -iDWin0; @is <= iDWin0; @is++)
                        {
                            var aiindex = ((iDWin + @is) * ihwl) + iDWin;
                            var ail = ((y + @is) * iWidth) + x;

                            for (var ir = -iDWin0; ir <= iDWin0; ir++)
                            {
                                var iindex = aiindex + ir;
                                var il = ail + ir;

                                fpCount[il]++;

                                for (var ii = 0; ii < iChannels; ii++)
                                {
                                    fpO[ii][il] += fpODenoised[ii][iindex] / fTotalWeight;
                                }
                            }
                        }
                    }
                }
            });

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
        }

        private void wxFillExpLut(float[] lut, int size)
        {
            for (int i = 0; i < size; i++)
            {
                lut[i] = (float)Math.Exp(-i / LUTPRECISION);
            }
        }

        private float wxSLUT(float dif, float[] lut)
        {
            if (dif >= LUTMAXM1)
            {
                return 0.0f;
            }

            var x = (int)Math.Floor(dif * LUTPRECISION);

            var y1 = lut[x];
            var y2 = lut[x + 1];

            return y1 + ((y2 - y1) * ((dif * LUTPRECISION) - x));
        }

        private float fiL2FloatDist(float** u0, float** u1, int i0, int j0, int i1, int j1, int radius, int channels, int width0, int width1)
        {
            var dif = 0.0f;

            for (var ii = 0; ii < channels; ii++)
            {
                dif += this.fiL2FloatDist(u0[ii], u1[ii], i0, j0, i1, j1, radius, width0, width1);
            }

            return dif;
        }

        private float fiL2FloatDist(float* u0, float* u1, int i0, int j0, int i1, int j1, int radius, int width0, int width1)
        {
            var dist = 0.0f;
            for (int s = -radius; s <= radius; s++)
            {
                var l = ((j0 + s) * width0) + (i0 - radius);
                var ptr0 = &u0[l];

                l = ((j1 + s) * width1) + (i1 - radius);
                var ptr1 = &u1[l];

                for (var r = -radius; r <= radius; r++, ptr0++, ptr1++)
                {
                    var dif = *ptr0 - *ptr1;
                    dist += dif * dif;
                }
            }

            return dist;
        }
    }
}