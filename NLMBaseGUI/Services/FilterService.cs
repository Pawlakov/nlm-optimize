namespace NLMBaseGUI.Services
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using NLMBaseGUI.NLM;

    public class FilterService
    {
        private IImplementation implementation;

        public FilterService()
        {
            this.implementation = new DefaultImplementation();
        }

        public Bitmap MakeFiltered(Bitmap input, int sigma)
        {
            
        }
    }
}
