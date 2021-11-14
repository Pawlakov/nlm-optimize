namespace NLMBaseGUI.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using NLMBaseGUI.Models;

    public interface ISession
    {
        Task<(Bitmap, FilteringStatsModel)> Run(Bitmap? raw);

        Task Cancel();
    }
}
