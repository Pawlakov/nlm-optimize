namespace NLMBase
{
    public unsafe delegate void DenoiseFunction(byte* inputPointer, byte* outputPointer, int h);
}