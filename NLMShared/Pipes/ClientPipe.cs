namespace NLMShared.Pipes
{
    using System;
    using System.IO.Pipes;

    public class ClientPipe
        : BasicPipe
    {
        private NamedPipeClientStream clientPipeStream;

        public ClientPipe(string serverName, string pipeName, Action<BasicPipe> asyncReaderStart)
        {
            this.AsyncReaderStart = asyncReaderStart;
            this.clientPipeStream = new NamedPipeClientStream(serverName, pipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
            this.PipeStream = this.clientPipeStream;
        }

        public void Connect()
        {
            this.clientPipeStream.Connect();
            this.AsyncReaderStart(this);
        }
    }
}
