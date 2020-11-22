using System;

namespace NLMBase
{
    public interface IImplementation3D : IDisposable
    {
        Denoise3DFunction Denoise { get; }
    }
}