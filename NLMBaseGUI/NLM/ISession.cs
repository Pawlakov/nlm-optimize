namespace NLMBaseGUI.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public interface ISession
    {
        Task Run();

        Task Cancel();
    }
}
