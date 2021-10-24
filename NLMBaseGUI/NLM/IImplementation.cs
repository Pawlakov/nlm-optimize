namespace NLMBaseGUI.NLM
{
    using System;

    public interface IImplementation 
        : IDisposable
    {
        string Name { get; }

        DenoiseFunction Denoise { get; }
    }
}