namespace NLMShared.Pipes
{
    using System;
    using System.Collections.Generic;
    using System.IO.Pipes;
    using System.Linq;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.Text;
    using System.Threading.Tasks;

    public abstract class BasicPipe
    {
        public event EventHandler<PipeEventArgs> DataReceived;

        protected PipeStream PipeStream { get; set; }

        protected Action<BasicPipe> AsyncReaderStart { get; set; }

        public void Close()
        {
            this.PipeStream.WaitForPipeDrain();
            this.PipeStream.Close();
            this.PipeStream.Dispose();
            this.PipeStream = null;
        }

        public void StartObjectReaderAsync()
        {
            try
            {
                var f = new BinaryFormatter();
                var messageReceived = f.Deserialize(this.PipeStream);
                this.DataReceived?.Invoke(this, new PipeEventArgs(messageReceived));
            }
            catch (Exception exception)
            {
                this.DataReceived?.Invoke(this, new PipeEventArgs(exception));
            }
        }

        public void Flush()
        {
            this.PipeStream.Flush();
        }

        public Task WriteObject(object obj)
        {
            var f = new BinaryFormatter();
            return Task.Run(() => f.Serialize(this.PipeStream, obj));
        }
    }
}
