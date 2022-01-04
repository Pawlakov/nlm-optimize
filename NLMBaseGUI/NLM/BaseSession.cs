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
    using SkiaSharp;

    public abstract class BaseSession
        : ISession
    {
        public abstract Task Cancel();

        public abstract Task<(SKBitmap, FilteringStatsModel)> Run(SKBitmap raw);
    }
}
