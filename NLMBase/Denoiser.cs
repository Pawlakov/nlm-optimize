namespace NLMBase
{
    using MersenneTwister;
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Runtime.InteropServices;
    using System.Threading;

    public unsafe class Denoiser
    {
        private readonly int width;

        private readonly int height;

        private readonly int stride;

        private readonly int length;

        private readonly int channels;

        private readonly byte[] inputArray;

        private readonly PixelFormat pixelFormat;

        private readonly IImplementation library;

        public Denoiser(Bitmap input, IImplementation library)
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
            this.inputArray = new byte[this.length];
            Marshal.Copy(inputOrigin, inputArray, 0, this.length);
            input.UnlockBits(inputData);
            this.library = library;
        }

        public long Work(int sigma, out Bitmap noisy, out Bitmap result, out float? mseNoisy, out float? mseResult, out float? ssimNoisy, out float? ssimResult)
        {
            var inputChannels = UnwrapChannels(this.inputArray);

            var noisyChannels = MakeEmptyChannels();
            this.Noise(inputChannels, noisyChannels, sigma);

            var resultChannels = MakeEmptyChannels();

            Console.WriteLine("Timer START");
            var watch = Stopwatch.StartNew();
            this.Denoise(noisyChannels, resultChannels, sigma);
            watch.Stop();
            Console.WriteLine("Timer STOP");

            noisy = new Bitmap(this.width, this.height, pixelFormat);
            //var noisyData = noisy.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            //var noisyOrigin = noisyData.Scan0;
            var noisyArray = WrapChannels(noisyChannels);
            //Marshal.Copy(noisyArray, 0, noisyOrigin, this.length);
            //noisy.UnlockBits(noisyData);
            this.WriteBitemapTheDumbWay(noisy, noisyArray);

            result = new Bitmap(this.width, this.height, pixelFormat);
            //var resultData = result.LockBits(new Rectangle(0, 0, this.width, this.height), ImageLockMode.ReadOnly, noisy.PixelFormat);
            //var resultOrigin = resultData.Scan0;
            var resultArray = WrapChannels(resultChannels);
            //Marshal.Copy(resultArray, 0, resultOrigin, this.length);
            //result.UnlockBits(resultData);
            this.WriteBitemapTheDumbWay(result, resultArray);

            try
            {
                mseResult = this.MSE(this.inputArray, resultArray);
            }
            catch
            {
                mseResult = null;
            }

            try
            {
                mseNoisy = this.MSE(this.inputArray, noisyArray);
            }
            catch
            {
                mseNoisy = null;
            }

            try
            {
                ssimResult = this.SSIM(this.inputArray, resultArray);
            }
            catch
            {
                ssimResult = null;
            }

            try
            {
                ssimNoisy = this.SSIM(this.inputArray, noisyArray);
            }
            catch
            {
                ssimNoisy = null;
            }

            return watch.ElapsedMilliseconds;
        }

        private void Noise(float[] inputPointer, float[] outputPointer, int sigma)
        {
            var random = Randoms.Create(DateTime.Now.Millisecond + Thread.CurrentThread.ManagedThreadId, RandomType.FastestDouble);
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

            fixed (float* inputFlatPointer = &inputArray[0], outputFlatPointer = &outputArray[0])
            {
                var fpI = new float*[this.channels];
                var fpO = new float*[this.channels];
                for (int ii = 0; ii < this.channels; ii++)
                {
                    fpI[ii] = &inputFlatPointer[ii * this.width * this.height];
                    fpO[ii] = &outputFlatPointer[ii * this.width * this.height];
                }

                fixed (float** inputPointer = &fpI[0], outputPointer = &fpO[0])
                {
                    this.library.Denoise(win, bloc, sigma, fFiltPar, inputPointer, outputPointer, this.channels, this.width, this.height);
                }
            }
        }

        private float MSE(byte[] firstArray, byte[] secondArray)
        {
            var size = this.width * this.height * this.channels;
            var sum = 0.0f;
            for (var i = 0; i < size; ++i)
            {
                var distance = (firstArray[i] - secondArray[i]) / 255.0f;
                sum += (distance * distance);
            }

            return sum / size;
        }

        private float SSIM(byte[] firstArray, byte[] secondArray)
        {
            var channelSize = this.height * this.width;
            var firstSingleArray = new float[channelSize];
            var secondSingleArray = new float[channelSize];

            for (var i = 0; i < channelSize; ++i)
            {
                for (var j = 0; j < this.channels; ++j)
                {
                    firstSingleArray[i] = 0.3f * firstArray[i + channelSize + channelSize] + 0.59f * firstArray[i + channelSize] + 0.11f * firstArray[i];
                    secondSingleArray[i] = 0.3f * secondArray[i + channelSize + channelSize] + 0.59f * secondArray[i + channelSize] + 0.11f * secondArray[i];
                }
            }

            var ssim = new SSIM();
            return ssim.ComputeSSIM(firstSingleArray, secondSingleArray, this.width, this.height);
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

        private void WriteBitemapTheDumbWay(Bitmap bitmap, byte[] bytesWrapped)
        {
            for (var x = 0; x < this.width; ++x)
            {
                for (var y = 0; y < this.height; ++y)
                {
                    switch (this.channels)
                    {
                        case 1:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[y * this.stride + x * 1 + 0], bytesWrapped[y * this.stride + x * 1 + 0], bytesWrapped[y * this.stride + x] * 1 + 0));
                            break;
                        case 2:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[y * this.stride + x * 2 + 1], bytesWrapped[y * this.stride + x * 2 + 0], bytesWrapped[y * this.stride + x * 2 + 0], bytesWrapped[y * this.stride + x * 2 + 0]));
                            break;
                        case 3:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[y * this.stride + x * 3 + 2], bytesWrapped[y * this.stride + x * 3 + 1], bytesWrapped[y * this.stride + x * 3 + 0]));
                            break;
                        case 4:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[y * this.stride + x * 4 + 3], bytesWrapped[y * this.stride + x * 4 + 2], bytesWrapped[y * this.stride + x * 4 + 1], bytesWrapped[y * this.stride + x * 4 + 0]));
                            break;
                    }
                }
            }
        }
    
        public static bool CheckInputFile(string inputFilePath)
        {
            try
            {
                var bitmap = new Bitmap(inputFilePath);
                var channels = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
                if (channels == 1 || channels == 3)
                {
                    return true;
                }
            }
            catch { }

            return false;
        }
    
        public static bool CheckSigma(string value)
        {
            try
            {
                var number = int.Parse(value);
                if (number >= 1 && number <= 100)
                {
                    return true;
                }
            }
            catch { }

            return false;
        }
    }
}