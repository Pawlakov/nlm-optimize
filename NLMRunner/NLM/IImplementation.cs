namespace NLMRunner.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using NLMShared.Models;

    public interface IImplementation
        : IDisposable
    {
        void RunDenoise(float[] inputArray, float[] outputArray, NLMParamsModel nlmParams, int channels, int width, int height);
    }
}
