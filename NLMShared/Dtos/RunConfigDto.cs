namespace NLMShared.Dtos
{
    using System;
    using NLMShared.Models;

    [Serializable]
    public class RunConfigDto
    {
        public NLMParamsModel Params { get; set; }

        public byte[] InputFile { get; set; }

        public string LibraryPath { get; set; }
    }
}
