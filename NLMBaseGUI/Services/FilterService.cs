namespace NLMBaseGUI.Services
{
    using NLMBaseGUI.Models;
    using NLMBaseGUI.NLM;
    using SkiaSharp;

    public unsafe class FilterService
    {
        public ISession SetUp(ImplementationModel library, SKBitmap noisy, int sigma, int windowRadius, int blockRadius, float filterParam)
        {
            return new ExternalSession(sigma, windowRadius, blockRadius, filterParam, noisy, library.File?.FullName);
        }
    }
}
