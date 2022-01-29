namespace NLMBaseGUI.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using NLMShared.Models;
    using SkiaSharp;

    public interface ISession
    {
        Task<(SKBitmap, FilteringStatsModel)> Run(SKBitmap raw);

        Task Cancel();
    }
}
