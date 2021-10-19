namespace NLMBaseGUI.Services
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Threading.Tasks;
    using NLMBaseGUI.Helpers;
    using NLMBaseGUI.NLM;

    public unsafe class FilterService
    {
        private IImplementation library;

        public FilterService()
        {
            this.library = new DefaultImplementation();
        }

        public Bitmap MakeFiltered(Bitmap noisy, int sigma)
        {
            var width = Math.Min(noisy.Width, noisy.Width);
            var height = Math.Min(noisy.Height, noisy.Height);
            var noisyData = noisy.LockBits(
                new Rectangle(0, 0, noisy.Width, noisy.Height),
                ImageLockMode.ReadOnly,
                noisy.PixelFormat);
            var stride = noisyData.Stride;
            var pixelFormat = noisyData.PixelFormat;
            var channels = Image.GetPixelFormatSize(pixelFormat) / 8;
            var noisyOrigin = noisyData.Scan0;
            var length = Math.Abs(noisyData.Stride) * noisyData.Height;
            var noisyArray = new byte[length];
            Marshal.Copy(noisyOrigin, noisyArray, 0, length);
            noisy.UnlockBits(noisyData);

            var noisyChannels = BitmapHelpers.UnwrapChannels(noisyArray, channels, width, height, stride);

            var filteredChannels = BitmapHelpers.MakeEmptyChannels(channels, width, height);
            this.Denoise(noisyChannels, filteredChannels, sigma, channels, width, height);

            var filtered = new Bitmap(width, height, pixelFormat);
            var filteredArray = BitmapHelpers.WrapChannels(filteredChannels, channels, width, height, length, stride);
            BitmapHelpers.WriteBitemapTheDumbWay(filtered, filteredArray, channels, width, height, stride);

            return filtered;
        }

        private void Denoise(float[] inputArray, float[] outputArray, int sigma, int channels, int width, int height)
        {
            var win = 0;
            var bloc = 0;
            var fFiltPar = 0.0f;
            if (channels < 3)
            {
                if (sigma > 0 && sigma <= 15)
                {
                    win = 1;
                    bloc = 10;
                    fFiltPar = 0.4f;
                }
                else if (sigma > 15 && sigma <= 30)
                {
                    win = 2;
                    bloc = 10;
                    fFiltPar = 0.4f;
                }
                else if (sigma > 30 && sigma <= 45)
                {
                    win = 3;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else if (sigma > 45 && sigma <= 75)
                {
                    win = 4;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else if (sigma <= 100)
                {
                    win = 5;
                    bloc = 17;
                    fFiltPar = 0.30f;
                }
                else
                {
                    throw new Exception("sigma > 100");
                }
            }
            else
            {
                if (sigma > 0 && sigma <= 25)
                {
                    win = 1;
                    bloc = 10;
                    fFiltPar = 0.55f;
                }
                else if (sigma > 25 && sigma <= 55)
                {
                    win = 2;
                    bloc = 17;
                    fFiltPar = 0.4f;
                }
                else if (sigma <= 100)
                {
                    win = 3;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else
                {
                    throw new Exception("sigma > 100");
                }
            }

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
                    this.library.Denoise(win, bloc, sigma, fFiltPar, inputPointer, outputPointer, channels, width, height);
                }
            }
        }
    }
}
