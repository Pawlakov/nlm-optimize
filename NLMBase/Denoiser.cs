namespace NLMBase
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;

    public unsafe class Denoiser : IDisposable
    {
        private readonly int width;

        private readonly int height;

        private readonly int inputStride;

        private readonly byte* inputOrigin;

        private readonly PixelFormat inputPixelFormat;

        private readonly int inputBytesPerPixel;

        private readonly Action disposeAction;

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
            this.inputBytesPerPixel = Image.GetPixelFormatSize(inputData.PixelFormat) / 8;
            this.inputOrigin = (byte*)inputData.Scan0.ToPointer();
            this.disposeAction = () =>
                {
                    input.UnlockBits(inputData);
                };
            this.library = library;
        }

        public long Denoise(int sigma, out Bitmap noisy, out Bitmap result)
        {
            noisy = new Bitmap(this.width, this.height, inputPixelFormat);
            var noisyData = noisy.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            var noisyOrigin = (byte*)noisyData.Scan0.ToPointer();

            this.Noise(this.inputOrigin, noisyOrigin, this.height * noisyData.Stride, sigma);

            result = new Bitmap(this.width, this.height, inputPixelFormat);
            var resultData = result.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            var resultOrigin = (byte*)resultData.Scan0.ToPointer();

            var watch = Stopwatch.StartNew();
            this.library.Denoise(noisyOrigin, resultOrigin, this.height * resultData.Stride, sigma);
            watch.Stop();

            Console.WriteLine($"Color [R={noisyOrigin[0]}, G={noisyOrigin[1]}, B={noisyOrigin[2]}]");
            noisy.UnlockBits(noisyData);
            Console.WriteLine(noisy.GetPixel(0, 0));
            result.UnlockBits(resultData);
            return watch.ElapsedTicks;
        }

        public void Dispose()
        {
            this.disposeAction();
        }

        private void Noise(byte* inputPointer, byte* outputPointer, int length, int sigma)
        {
            var random = new Random();
            for (var i = 0; i < length; ++i)
            {
                //var a = random.NextDouble();
                //var b = random.NextDouble();
                //var noise = (double)(sigma) * Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b);
                //outputPointer[i] = (byte)((double)inputPointer[i] + noise);
                outputPointer[i] = inputPointer[i];
            }
        }
    }
}