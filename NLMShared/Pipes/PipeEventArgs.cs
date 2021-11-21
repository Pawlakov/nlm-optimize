namespace NLMShared.Pipes
{
    using System;
    using System.Text;

    public class PipeEventArgs
    {
        public PipeEventArgs(byte[] data)
        {
            this.Data = data;
            this.String = Encoding.UTF8.GetString(data).TrimEnd('\0');
        }

        public byte[] Data { get; protected set; }

        public string String { get; protected set; }

        public int Len => this.Data?.Length ?? 0;
    }
}
