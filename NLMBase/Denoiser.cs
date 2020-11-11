namespace NLMBase
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Runtime.InteropServices;

    public unsafe class Denoiser
    {
        private readonly int width;

        private readonly int height;

        private readonly int stride;

        private readonly int length;

        private readonly int channels;

        private readonly float[] inputChannels;

        private readonly PixelFormat pixelFormat;

        private readonly Implementation library;

        public Denoiser(Bitmap input, Implementation library)
        {
            this.width = Math.Min(input.Width, input.Width);
            this.height = Math.Min(input.Height, input.Height);
            var inputData = input.LockBits(
                new Rectangle(0, 0, input.Width, input.Height),
                ImageLockMode.ReadOnly,
                input.PixelFormat);
            this.stride = inputData.Stride;
            this.pixelFormat = inputData.PixelFormat;
            this.channels = Image.GetPixelFormatSize(this.pixelFormat) / 8;
            var inputOrigin = inputData.Scan0;
            this.length = Math.Abs(inputData.Stride) * inputData.Height;
            var inputArray = new byte[this.length];
            Marshal.Copy(inputOrigin, inputArray, 0, this.length);
            input.UnlockBits(inputData);
            this.inputChannels = UnwrapChannels(inputArray);
            this.library = library;
        }

        public long Work(int sigma, out Bitmap noisy, out Bitmap result)
        {
            var noisyChannels = MakeEmptyChannels();
            this.Noise(this.inputChannels, noisyChannels, sigma);

            var resultChannels = MakeEmptyChannels();

            var watch = Stopwatch.StartNew();
            this.Denoise(noisyChannels, resultChannels, sigma);
            watch.Stop();

            noisy = new Bitmap(this.width, this.height, pixelFormat);
            var noisyData = noisy.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            var noisyOrigin = noisyData.Scan0;
            var noisyArray = WrapChannels(noisyChannels);
            Marshal.Copy(noisyArray, 0, noisyOrigin, this.length);
            noisy.UnlockBits(noisyData);

            result = new Bitmap(this.width, this.height, pixelFormat);
            var resultData = result.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            var resultOrigin = resultData.Scan0;
            var resultArray = WrapChannels(resultChannels);
            Marshal.Copy(resultArray, 0, resultOrigin, this.length);
            result.UnlockBits(resultData);

            return watch.ElapsedTicks;
        }

        private void Noise(float[] inputPointer, float[] outputPointer, int sigma)
        {
            var random = new Random();
            var size = this.width * this.height * this.channels;
            for (var i = 0; i < size; ++i)
            {
                var a = random.NextDouble();
                var b = random.NextDouble();
                var noise = (float)(sigma * Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b));
                outputPointer[i] = (float)inputPointer[i] + noise;
            }
        }

        private void Denoise(float[] inputArray, float[] outputArray, int sigma)
        {
            var win = 0;
            var bloc = 0;
            var fFiltPar = 0.0f;
            if (this.channels < 3)
            {
                if (sigma > 0 && sigma <= 15)
                {
                    win = 1;
                    bloc = 10;
                    fFiltPar = 0.4f;
                }
                else if (sigma > 15 && sigma <= 30)
                {
                    win = 2;
                    bloc = 10;
                    fFiltPar = 0.4f;
                }
                else if (sigma > 30 && sigma <= 45)
                {
                    win = 3;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else if (sigma > 45 && sigma <= 75)
                {
                    win = 4;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else if (sigma <= 100)
                {
                    win = 5;
                    bloc = 17;
                    fFiltPar = 0.30f;
                }
                else
                {
                    throw new Exception("sigma > 100");
                }
            }
            else
            {
                if (sigma > 0 && sigma <= 25)
                {
                    win = 1;
                    bloc = 10;
                    fFiltPar = 0.55f;
                }
                else if (sigma > 25 && sigma <= 55)
                {
                    win = 2;
                    bloc = 17;
                    fFiltPar = 0.4f;
                }
                else if (sigma <= 100)
                {
                    win = 3;
                    bloc = 17;
                    fFiltPar = 0.35f;
                }
                else
                {
                    throw new Exception("sigma > 100");
                }
            }

            fixed (float* inputPointer = &inputArray[0], outputPointer = &outputArray[0])
            {
                this.library.Denoise(win, bloc, sigma, fFiltPar, inputPointer, outputPointer, this.channels, this.width, this.height);
            }
        }

        private float[] UnwrapChannels(byte[] input)
        {
            var output = new float[this.channels * this.width * this.height];
            for (var i = 0; i < this.channels; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        output[this.width * (this.height * i + j) + k] = input[j * this.stride + k * this.channels + i];
                    }
                }
            }

            return output;
        }

        private byte[] WrapChannels(float[] input)
        {
            var output = new byte[this.length];
            for (var i = 0; i < this.channels; ++i)
            {
                for (var j = 0; j < this.height; ++j)
                {
                    for (var k = 0; k < this.width; ++k)
                    {
                        var value = input[this.width * (this.height * i + j) + k];
                        output[j * this.stride + k * this.channels + i] = (byte)Math.Clamp(Math.Floor(value + 0.5), 0.0, 255.0);
                    }
                }
            }

            return output;
        }

        private float[] MakeEmptyChannels()
        {
            var resultChannels = new float[this.channels * this.width * this.height];

            return resultChannels;
        }
    }
}