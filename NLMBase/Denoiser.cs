namespace NLMBase
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Runtime.InteropServices;

    public unsafe class Denoiser
    {
        private readonly int width;

        private readonly int height;

        private readonly int inputStride;

        private readonly byte[] inputArray;

        private readonly PixelFormat inputPixelFormat;

        private readonly Implementation library;

        public Denoiser(Bitmap input, Implementation library)
        {
            this.width = Math.Min(input.Width, input.Width);
            this.height = Math.Min(input.Height, input.Height);
            var inputData = input.LockBits(
                new Rectangle(0, 0, input.Width, input.Height),
                ImageLockMode.ReadOnly,
                input.PixelFormat);
            this.inputStride = inputData.Stride;
            this.inputPixelFormat = inputData.PixelFormat;
            var inputOrigin = inputData.Scan0;
            var inputLength = Math.Abs(inputData.Stride) * inputData.Height;
            this.inputArray = new byte[inputLength];
            Marshal.Copy(inputOrigin, this.inputArray, 0, inputLength);
            input.UnlockBits(inputData);
            this.library = library;
        }

        public long Denoise(int sigma, out Bitmap noisy, out Bitmap result)
        {
            noisy = new Bitmap(this.width, this.height, inputPixelFormat);
            var noisyData = noisy.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            var noisyOrigin = noisyData.Scan0;
            var noisyLength = Math.Abs(noisyData.Stride) * noisyData.Height;
            var noisyArray = new byte[noisyLength];
            Marshal.Copy(noisyOrigin, noisyArray, 0, noisyLength);

            this.Noise(this.inputArray, noisyArray, noisyLength, sigma);

            result = new Bitmap(this.width, this.height, inputPixelFormat);
            var resultData = result.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            var resultOrigin = resultData.Scan0;
            var resultLength = Math.Abs(resultData.Stride) * resultData.Height;
            var resultArray = new byte[resultLength];
            Marshal.Copy(resultOrigin, resultArray, 0, resultLength);

            var watch = Stopwatch.StartNew();
            this.library.Denoise(noisyArray, resultArray, resultLength, sigma);
            watch.Stop();

            Marshal.Copy(noisyArray, 0, noisyOrigin, noisyLength);
            Marshal.Copy(resultArray, 0, resultOrigin, resultLength);

            noisy.UnlockBits(noisyData);
            result.UnlockBits(resultData);
            return watch.ElapsedTicks;
        }

        private void Noise(byte[] inputPointer, byte[] outputPointer, int length, int sigma)
        {
            var random = new Random();
            for (var i = 0; i < length; ++i)
            {
                var a = random.NextDouble();
                var b = random.NextDouble();
                var noise = (double)(sigma) * Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b);
                var value = ((double)inputPointer[i] + noise);
                outputPointer[i] = (byte)Math.Clamp(Math.Floor(value + 0.5), 0.0, 255.0);
            }
        }
    }
}