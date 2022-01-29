namespace NLMBaseGUI.Models
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public class ImplementationModel
    {
        public ImplementationModel(FileInfo file)
        {
            this.File = file;
        }

        public string Name => this.File == null ? "Domyślna" : this.File.Name;

        public FileInfo File { get; set; }
    }
}
