namespace NLMShared.Dtos
{
    using System;

    public class RunResultDto
    {
        public long Time { get; set; }

        /*public string OutputPath { get; set; }*/
        public string OutputFile { get; set; }

        public Exception? Exception { get; set; }
    }
}
