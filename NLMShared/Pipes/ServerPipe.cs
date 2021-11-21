namespace NLMShared.Pipes
{
    using System;
    using System.IO.Pipes;

    public class ServerPipe
        : BasicPipe
    {
        private readonly NamedPipeServerStream serverPipeStream;

        public ServerPipe(string pipeName, Action<BasicPipe> asyncReaderStart)
        {
            this.AsyncReaderStart = asyncReaderStart;
            this.PipeName = pipeName;

            this.serverPipeStream = new NamedPipeServerStream(
                pipeName,
                PipeDirection.InOut,
                NamedPipeServerStream.MaxAllowedServerInstances,
                PipeTransmissionMode.Byte,
                PipeOptions.Asynchronous);

            this.PipeStream = this.serverPipeStream;
            this.serverPipeStream.BeginWaitForConnection(new AsyncCallback(this.PipeConnected), null);
        }

        public event EventHandler<EventArgs> Connected;

        protected string PipeName { get; set; }

        protected void PipeConnected(IAsyncResult ar)
        {
            this.serverPipeStream.EndWaitForConnection(ar);
            this.Connected?.Invoke(this, new EventArgs());
            this.AsyncReaderStart(this);
        }
    }
}
