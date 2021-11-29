namespace NLMBaseGUI.Services
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;
    using NLMBaseGUI.Models;
    using NLMBaseGUI.NLM;
    using NLMShared.Helpers;
    using NLMShared.Models;
    using NLMShared.NLM;

    public unsafe class FilterService
    {
        public ISession SetUp(ImplementationModel library, Bitmap noisy, int sigma, int windowRadius, int blockRadius, float filterParam)
        {
            return new ExternalSession(sigma, windowRadius, blockRadius, filterParam, noisy, library.File?.FullName);
        }
    }
}
