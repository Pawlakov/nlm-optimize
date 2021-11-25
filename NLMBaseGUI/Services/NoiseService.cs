namespace NLMBaseGUI.Services
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;
    using MersenneTwister;
    using NLMBaseGUI.Models;
    using NLMShared.Helpers;

    public class NoiseService
    {
        public (Bitmap, FilteringStatsModel) MakeNoisy(Bitmap input, int sigma)
        {
            var width = Math.Min(input.Width, input.Width);
            var height = Math.Min(input.Height, input.Height);
            var inputData = input.LockBits(
                new Rectangle(0, 0, input.Width, input.Height),
                ImageLockMode.ReadOnly,
                input.PixelFormat);
            var stride = inputData.Stride;
            var pixelFormat = inputData.PixelFormat;
            var channels = Image.GetPixelFormatSize(pixelFormat) / 8;
            var inputOrigin = inputData.Scan0;
            var length = Math.Abs(inputData.Stride) * inputData.Height;
            var inputArray = new byte[length];
            Marshal.Copy(inputOrigin, inputArray, 0, length);
            input.UnlockBits(inputData);

            var inputChannels = BitmapHelpers.UnwrapChannels(inputArray, channels, width, height, stride);
            var noisyChannels = BitmapHelpers.MakeEmptyChannels(channels, width, height);

            var watch = Stopwatch.StartNew();
            this.Noise(inputChannels, noisyChannels, sigma, channels, width, height);
            watch.Stop();

            var noisy = new Bitmap(width, height, pixelFormat);
            var noisyArray = BitmapHelpers.WrapChannels(noisyChannels, channels, width, height, length, stride);
            BitmapHelpers.WriteBitemapTheDumbWay(noisy, noisyArray, channels, width, height, stride);

            float? mseNoisy;
            try
            {
                mseNoisy = BitmapHelpers.CalculateMSE(inputArray, noisyArray, width, height, channels);
            }
            catch
            {
                mseNoisy = null;
            }

            float? ssimNoisy;
            try
            {
                ssimNoisy = BitmapHelpers.CalculateSSIM(inputArray, noisyArray, width, height, channels);
            }
            catch
            {
                ssimNoisy = null;
            }

            var stats = new FilteringStatsModel
            {
                Time = TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds),
                MSE = mseNoisy,
                SSIM = ssimNoisy,
            };

            return (noisy, stats);
        }

        private void Noise(float[] inputPointer, float[] outputPointer, int sigma, int channels, int width, int height)
        {
            var randomLock = new object();
            var random = Randoms.Create(DateTime.Now.Millisecond + Thread.CurrentThread.ManagedThreadId, RandomType.FastestDouble);
            var size = width * height * channels;
            Parallel.For(0, size, (i, state) =>
            {
                double a, b;
                lock (randomLock)
                {
                    a = random.NextDouble();
                    b = random.NextDouble();
                }

                var noise = (float)(sigma * Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b));
                outputPointer[i] = (float)inputPointer[i] + noise;
            });
        }
    }
}
