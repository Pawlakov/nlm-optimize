namespace NLMRunner.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public interface IImplementation
        : IDisposable
    {
        void RunDenoise(float[] inputArray, float[] outputArray, int sigma, int channels, int width, int height);
    }
}
