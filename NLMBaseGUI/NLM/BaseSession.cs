namespace NLMBaseGUI.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using NLMBaseGUI.Models;
    using NLMShared.Dtos;
    using NLMShared.Helpers;
    using NLMShared.Models;

    public abstract class BaseSession
        : ISession
    {
        public abstract Task Cancel();

        public abstract Task<(Bitmap, FilteringStatsModel)> Run(Bitmap? raw);

        protected FilteringStatsModel CalculateStats(Bitmap? raw, Bitmap altered, long time)
        {
            var mseResult = (float?)null;
            var ssimResult = (float?)null;

            if (raw != null)
            {
                var rawModel = BitmapModel.Create(raw);
                var alteredModel = BitmapModel.Create(altered);

                var rawArray = BitmapHelpers.WrapChannels(rawModel.Data, rawModel.Channels, rawModel.Width, rawModel.Height, rawModel.Length, rawModel.Stride);
                var alteredArray = BitmapHelpers.WrapChannels(alteredModel.Data, alteredModel.Channels, alteredModel.Width, alteredModel.Height, alteredModel.Length, alteredModel.Stride);

                if (alteredModel.Width == rawModel.Width && alteredModel.Height == rawModel.Height && alteredModel.Channels == rawModel.Channels)
                {
                    try
                    {
                        mseResult = BitmapHelpers.CalculateMSE(rawArray, alteredArray, alteredModel.Width, alteredModel.Height, alteredModel.Channels);
                    }
                    catch
                    {
                        mseResult = null;
                    }

                    try
                    {
                        ssimResult = BitmapHelpers.CalculateSSIM(rawArray, alteredArray, alteredModel.Width, alteredModel.Height, alteredModel.Channels);
                    }
                    catch
                    {
                        ssimResult = null;
                    }
                }
            }

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
