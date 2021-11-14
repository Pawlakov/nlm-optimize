namespace NLMShared.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using NLMShared.Models;

    public abstract class BaseImplementation
        : IDisposable
    {
        public abstract void Dispose();

        protected NLMParamsModel MakeParams(int sigma, int channels)
        {
            var win = 0;
            var bloc = 0;
            var fFiltPar = 0.0f;
            if (channels < 3)
            {
                if (sigma > 0 && sigma <= 15)
                {
                    win = 1;
                    bloc = 10;
                    fFiltPar = 0.4f;
                }
                else if (sigma > 15 && sigma <= 30)
                {
                    win = 2;
                    bloc = 10;
                    fFiltPar = 0.4f;
                }
                else if (sigma > 30 && sigma <= 45)
                {
                    win = 3;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else if (sigma > 45 && sigma <= 75)
                {
                    win = 4;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else if (sigma <= 100)
                {
                    win = 5;
                    bloc = 17;
                    fFiltPar = 0.30f;
                }
                else
                {
                    throw new Exception("sigma > 100");
                }
            }
            else
            {
                if (sigma > 0 && sigma <= 25)
                {
                    win = 1;
                    bloc = 10;
                    fFiltPar = 0.55f;
                }
                else if (sigma > 25 && sigma <= 55)
                {
                    win = 2;
                    bloc = 17;
                    fFiltPar = 0.4f;
                }
                else if (sigma <= 100)
                {
                    win = 3;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else
                {
                    throw new Exception("sigma > 100");
                }
            }

            return new NLMParamsModel
            {
                Bloc = bloc,
                FiltPar = fFiltPar,
                Win = win,
            };
        }
    }
}
