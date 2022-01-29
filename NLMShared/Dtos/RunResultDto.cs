namespace NLMShared.Dtos
{
    using System;

    [Serializable]
    public class RunResultDto
    {
        public long Time { get; set; }

        public byte[] OutputFile { get; set; }

        public Exception Exception { get; set; }
    }
}
