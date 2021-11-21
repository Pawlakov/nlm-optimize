namespace NLMShared.Pipes
{
    using System;
    using System.Collections.Generic;
    using System.IO.Pipes;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public abstract class BasicPipe
    {
        public event EventHandler<PipeEventArgs> DataReceived;

        public event EventHandler<EventArgs> PipeClosed;

        protected PipeStream PipeStream { get; set; }

        protected Action<BasicPipe> AsyncReaderStart { get; set; }

        public void Close()
        {
            this.PipeStream.WaitForPipeDrain();
            this.PipeStream.Close();
            this.PipeStream.Dispose();
            this.PipeStream = null;
        }

        /// <summary>
        /// Reads an array of bytes, where the first [n] bytes (based on the server's intsize) indicates the number of bytes to read to complete the packet.
        /// </summary>
        public void StartByteReaderAsync()
        {
            this.StartByteReaderAsync((b) => this.DataReceived?.Invoke(this, new PipeEventArgs(b)));
        }

        /// <summary>
        /// Reads an array of bytes, where the first [n] bytes (based on the server's intsize) indicates the number of bytes to read to complete the packet, and invokes the DataReceived event with a string converted from UTF8 of the byte array.
        /// </summary>
        public void StartStringReaderAsync()
        {
            this.StartByteReaderAsync((b) => this.DataReceived?.Invoke(this, new PipeEventArgs(b)));
        }

        public void Flush()
        {
            this.PipeStream.Flush();
        }

        public Task WriteString(string str)
        {
            return this.WriteBytes(Encoding.UTF8.GetBytes(str));
        }

        public Task WriteBytes(byte[] bytes)
        {
            var blength = BitConverter.GetBytes(bytes.Length);
            var bfull = blength.Concat(bytes).ToArray();

            return this.PipeStream.WriteAsync(bfull, 0, bfull.Length);
        }

        protected void StartByteReaderAsync(Action<byte[]> packetReceived)
        {
            int intSize = sizeof(int);
            byte[] bDataLength = new byte[intSize];

            this.PipeStream.ReadAsync(bDataLength, 0, intSize).ContinueWith(t =>
            {
                int len = t.Result;

                if (len == 0)
                {
                    this.PipeClosed?.Invoke(this, EventArgs.Empty);
                }
                else
                {
                    int dataLength = BitConverter.ToInt32(bDataLength, 0);
                    byte[] data = new byte[dataLength];

                    this.PipeStream.ReadAsync(data, 0, dataLength).ContinueWith(t2 =>
                    {
                        len = t2.Result;

                        if (len == 0)
                        {
                            this.PipeClosed?.Invoke(this, EventArgs.Empty);
                        }
                        else
                        {
                            packetReceived(data);
                            this.StartByteReaderAsync(packetReceived);
                        }
                    });
                }
            });
        }
    }
}
