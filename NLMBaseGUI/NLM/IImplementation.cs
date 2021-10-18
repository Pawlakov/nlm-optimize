namespace NLMBaseGUI.NLM
{
    using System;

    public interface IImplementation 
        : IDisposable
    {
        DenoiseFunction Denoise { get; }
    }
}