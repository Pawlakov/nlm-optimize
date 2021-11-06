using System;
using System.Text;

namespace NLMShared.Pipes
{
    public class PipeEventArgs
    {
        public byte[] Data { get; protected set; }
        public string String { get; protected set; }
        public int Len => Data?.Length ?? 0;

        public PipeEventArgs(byte[] data)
        {
            Data = data;
            String = Encoding.UTF8.GetString(data).TrimEnd('\0');
        }
    }
}
