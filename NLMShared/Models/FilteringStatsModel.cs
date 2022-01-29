namespace NLMShared.Models
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public class FilteringStatsModel
    {
        public TimeSpan Time { get; set; }

        public float? MSE { get; set; }

        public float? SSIM { get; set; }
    }
}
