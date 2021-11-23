namespace NLMShared.Dtos
{
    using System;

    [Serializable]
    public class RunConfigDto
    {
        public int Sigma { get; set; }

        public byte[] InputFile { get; set; }

        public string LibraryPath { get; set; }
    }
}
