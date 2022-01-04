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
    using NLMShared.Models;
    using SkiaSharp;

    public class NoiseService
    {
        public (SKBitmap, FilteringStatsModel) MakeNoisy(SKBitmap input, int sigma)
        {
            var inputChannels = BitmapModel.Create(input);
            var noisyChannels = BitmapModel.Create(inputChannels.Width, inputChannels.Height, inputChannels.ColorType, inputChannels.AlphaType);

            var watch = Stopwatch.StartNew();
            this.Noise(inputChannels, noisyChannels, sigma);
            watch.Stop();

            var noisy = noisyChannels.ToBitmap();

            var stats = BitmapHelpers.CalculateStats(input, noisy, watch.ElapsedMilliseconds);

            return (noisy, stats);
        }

        private void Noise(BitmapModel inputPointer, BitmapModel outputPointer, int sigma)
        {
            var random = Randoms.Create(DateTime.Now.Millisecond + Thread.CurrentThread.ManagedThreadId, RandomType.FastestDouble);
            for (var i = 0; i < inputPointer.Length; i++)
            {
                var a = random.NextDouble();
                var b = random.NextDouble();

                var noise = (float)(sigma * Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b));
                outputPointer.Data[i] = inputPointer.Data[i] + noise;
            }
        }
    }
}
