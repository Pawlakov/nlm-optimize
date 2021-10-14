namespace NLMBaseGUI.Services
{
    using MersenneTwister;
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;

    public class NoiseService
    {
        public Bitmap MakeNoisy(Bitmap input, int sigma)
        {
            var width = Math.Min(input.Width, input.Width);
            var height = Math.Min(input.Height, input.Height);
            var inputData = input.LockBits(
                new Rectangle(0, 0, input.Width, input.Height),
                ImageLockMode.ReadOnly,
                input.PixelFormat);
            var stride = inputData.Stride;
            var pixelFormat = inputData.PixelFormat;
            var channels = Image.GetPixelFormatSize(pixelFormat) / 8;
            var inputOrigin = inputData.Scan0;
            var length = Math.Abs(inputData.Stride) * inputData.Height;
            var inputArray = new byte[length];
            Marshal.Copy(inputOrigin, inputArray, 0, length);
            input.UnlockBits(inputData);

            var inputChannels = UnwrapChannels(inputArray, channels, width, height, stride);

            var noisyChannels = MakeEmptyChannels(channels, width, height);
            this.Noise(inputChannels, noisyChannels, sigma, channels, width, height);

            var noisy = new Bitmap(width, height, pixelFormat);
            //var noisyData = noisy.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            //var noisyOrigin = noisyData.Scan0;
            var noisyArray = WrapChannels(noisyChannels, channels, width, height, length, stride);
            //Marshal.Copy(noisyArray, 0, noisyOrigin, this.length);
            //noisy.UnlockBits(noisyData);
            this.WriteBitemapTheDumbWay(noisy, noisyArray, channels, width, height, stride);

            return noisy;
        }

        private void Noise(float[] inputPointer, float[] outputPointer, int sigma, int channels, int width, int height)
        {
            var random = Randoms.Create(DateTime.Now.Millisecond + Thread.CurrentThread.ManagedThreadId, RandomType.FastestDouble);
            var size = width * height * channels;
            for (var i = 0; i < size; ++i)
            {
                var a = random.NextDouble();
                var b = random.NextDouble();
                var noise = (float)(sigma * Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b));
                outputPointer[i] = (float)inputPointer[i] + noise;
            }
        }

        private float[] UnwrapChannels(byte[] input, int channels, int width, int height, int stride)
        {
            var output = new float[channels * width * height];
            for (var i = 0; i < channels; ++i)
            {
                for (var j = 0; j < height; ++j)
                {
                    for (var k = 0; k < width; ++k)
                    {
                        output[(width * ((height * i) + j)) + k] = input[(j * stride) + (k * channels) + i];
                    }
                }
            }

            return output;
        }

        private byte[] WrapChannels(float[] input, int channels, int width, int height, int length, int stride)
        {
            var output = new byte[length];
            for (var i = 0; i < channels; ++i)
            {
                for (var j = 0; j < height; ++j)
                {
                    for (var k = 0; k < width; ++k)
                    {
                        var value = input[(width * ((height * i) + j)) + k];
                        output[(j * stride) + (k * channels) + i] = (byte)Math.Clamp(Math.Floor(value + 0.5), 0.0, 255.0);
                    }
                }
            }

            return output;
        }

        private float[] MakeEmptyChannels(int channels, int width, int height)
        {
            var resultChannels = new float[channels * width * height];

            return resultChannels;
        }

        private void WriteBitemapTheDumbWay(Bitmap bitmap, byte[] bytesWrapped, int channels, int width, int height, int stride)
        {
            for (var x = 0; x < width; ++x)
            {
                for (var y = 0; y < height; ++y)
                {
                    switch (channels)
                    {
                        case 1:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 1) + 0], bytesWrapped[(y * stride) + (x * 1) + 0], (bytesWrapped[(y * stride) + x] * 1) + 0));
                            break;
                        case 2:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 2) + 1], bytesWrapped[(y * stride) + (x * 2) + 0], bytesWrapped[(y * stride) + (x * 2) + 0], bytesWrapped[(y * stride) + (x * 2) + 0]));
                            break;
                        case 3:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 3) + 2], bytesWrapped[(y * stride) + (x * 3) + 1], bytesWrapped[(y * stride) + (x * 3) + 0]));
                            break;
                        case 4:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 4) + 3], bytesWrapped[(y * stride) + (x * 4) + 2], bytesWrapped[(y * stride) + (x * 4) + 1], bytesWrapped[(y * stride) + (x * 4) + 0]));
                            break;
                    }
                }
            }
        }
    }
}
