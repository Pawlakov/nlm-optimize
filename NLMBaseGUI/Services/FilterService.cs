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
        public ISession SetUp(ImplementationModel library, Bitmap noisy, int sigma)
        {
            if (library.File != null)
            {
                return new ExternalSession(sigma, noisy, library.File.FullName);
            }
            else
            {
                return new InternalSession(sigma, noisy);
            }
        }
    }
}
