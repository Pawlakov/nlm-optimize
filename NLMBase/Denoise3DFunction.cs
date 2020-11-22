namespace NLMBase
{
    public unsafe delegate void Denoise3DFunction(int iDWin, int iDBloc, float fSigma, float fFiltPar, float** fpI, float** fpO, int iChannels, int iWidth, int iHeight, int iDepth);
}