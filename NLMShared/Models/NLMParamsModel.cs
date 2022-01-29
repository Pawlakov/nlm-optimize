namespace NLMShared.Models
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    [Serializable]
    public class NLMParamsModel
    {
        public int Sigma { get; set; }

        public int Win { get; set; }

        public int Bloc { get; set; }

        public float FiltPar { get; set; }
    }
}
