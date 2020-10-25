namespace NLMBase
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Threading.Tasks;

    public unsafe class Denoiser : IDisposable
    {
        private readonly int width;

        private readonly int height;

        private readonly int combinedStride;

        private readonly byte* combinedOrigin;

        private readonly int combinedBytesPerPixel;

        private readonly Action disposeAction;

        private readonly Implementation library;

        public Denoiser(Bitmap combined, Implementation library)
        {
            this.width = Math.Min(combined.Width, combined.Width);
            this.height = Math.Min(combined.Height, combined.Height);
            var combinedData = combined.LockBits(
                new Rectangle(0, 0, combined.Width, combined.Height),
                ImageLockMode.ReadOnly,
                combined.PixelFormat);
            this.combinedStride = combinedData.Stride;
            this.combinedBytesPerPixel = Program.GetBytesPerPixel(combinedData.PixelFormat);
            this.combinedOrigin = (byte*)combinedData.Scan0.ToPointer();
            this.disposeAction = () =>
                {
                    combined.UnlockBits(combinedData);
                };
            this.library = library;
        }

        public long Denoise(int h, out Bitmap result)
        {
            result = new Bitmap(this.width, this.height);
            var resultData = result.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, result.PixelFormat);
            var resultOrigin = (byte*)resultData.Scan0.ToPointer();
            var rectangle = new Rectangle(0, 0, this.width, this.height);

            var watch = Stopwatch.StartNew();
            this.library.Denoise(this.combinedOrigin + (rectangle.Y * resultData.Stride), resultOrigin + (rectangle.Y * resultData.Stride), rectangle.Height * resultData.Stride, h);
            watch.Stop();

            result.UnlockBits(resultData);
            return watch.ElapsedTicks;
        }

        public void Dispose()
        {
            this.disposeAction();
        }
    }
}