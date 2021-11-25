namespace NLMShared.Pipes
{
    using System;
    using System.Text;

    public class PipeEventArgs
    {
        public PipeEventArgs(object obj)
        {
            this.ObjectData = obj;
        }

        public PipeEventArgs(Exception exception)
        {
            this.Error = exception;
        }

        public object ObjectData { get; protected set; }

        public Exception Error { get; protected set; }
    }
}
