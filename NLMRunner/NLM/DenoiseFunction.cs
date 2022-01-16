namespace NLMRunner.NLM
{
    public unsafe delegate void DenoiseFunction(int iDWin, int iDBloc, float fSigma, float fFiltPar, float** fpI, float** fpO, int iChannels, int iWidth, int iHeight);
}