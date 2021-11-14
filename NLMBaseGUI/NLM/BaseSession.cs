namespace NLMBaseGUI.NLM
{
    using NLMBaseGUI.Models;
    using NLMShared.Dtos;
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public abstract class BaseSession
        : ISession
    {
        public abstract Task Cancel();

        public abstract Task<(Bitmap, FilteringStatsModel)> Run(Bitmap? raw);

        protected FilteringStatsModel CalculateStats(Bitmap? raw, Bitmap noisy, Bitmap filtered, long time)
        {
            var mseResult = (float?)null;
            var ssimResult = (float?)null;

            /*
            if (raw != null)
            {
                var rawWidth = Math.Min(raw.Width, raw.Width);
                var rawHeight = Math.Min(raw.Height, raw.Height);
                var rawData = raw.LockBits(
                    new Rectangle(0, 0, raw.Width, raw.Height),
                    ImageLockMode.ReadOnly,
                    raw.PixelFormat);
                var rawPixelFormat = rawData.PixelFormat;
                var rawChannels = Image.GetPixelFormatSize(rawPixelFormat) / 8;
                var rawOrigin = rawData.Scan0;
                var rawLength = Math.Abs(rawData.Stride) * rawData.Height;
                var rawArray = new byte[rawLength];
                Marshal.Copy(rawOrigin, rawArray, 0, rawLength);
                raw.UnlockBits(rawData);

                if (width == rawWidth && height == rawHeight && channels == rawChannels)
                {
                    try
                    {
                        mseResult = BitmapHelpers.CalculateMSE(rawArray, filteredArray, width, height, channels);
                    }
                    catch
                    {
                        mseResult = null;
                    }

                    try
                    {
                        ssimResult = BitmapHelpers.CalculateSSIM(rawArray, filteredArray, width, height, channels);
                    }
                    catch
                    {
                        ssimResult = null;
                    }
                }
            }
            */

            var stats = new FilteringStatsModel
            {
                Time = TimeSpan.FromMilliseconds(time),
                MSE = mseResult,
                SSIM = ssimResult,
            };

            return stats;
        }
    }
}
