using System;

namespace NLMBase
{
    public interface IImplementation : IDisposable
    {
        DenoiseFunction Denoise { get; }
    }
}