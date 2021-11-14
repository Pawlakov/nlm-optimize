namespace NLMShared.Helpers
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public static class BitmapHelpers
    {
        public static float[] UnwrapChannels(byte[] input, int channels, int width, int height, int stride)
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

        public static byte[] WrapChannels(float[] input, int channels, int width, int height, int length, int stride)
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

        public static float[] MakeEmptyChannels(int channels, int width, int height)
        {
            var resultChannels = new float[channels * width * height];

            return resultChannels;
        }

        public static void WriteBitemapTheDumbWay(Bitmap bitmap, byte[] bytesWrapped, int channels, int width, int height, int stride)
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

        public static float CalculateMSE(byte[] firstArray, byte[] secondArray, int width, int height, int channels)
        {
            var size = width * height * channels;
            var sum = 0.0f;
            for (var i = 0; i < size; ++i)
            {
                var distance = (firstArray[i] - secondArray[i]) / 255.0f;
                sum += distance * distance;
            }

            return sum / size;
        }

        public static float CalculateSSIM(byte[] firstArray, byte[] secondArray, int width, int height, int channels)
        {
            var channelSize = height * width;
            var firstSingleArray = new float[channelSize];
            var secondSingleArray = new float[channelSize];

            for (var i = 0; i < channelSize; ++i)
            {
                for (var j = 0; j < channels; ++j)
                {
                    firstSingleArray[i] = (0.3f * firstArray[i + channelSize + channelSize]) + (0.59f * firstArray[i + channelSize]) + (0.11f * firstArray[i]);
                    secondSingleArray[i] = (0.3f * secondArray[i + channelSize + channelSize]) + (0.59f * secondArray[i + channelSize]) + (0.11f * secondArray[i]);
                }
            }

            var ssim = new SSIM();
            return ssim.ComputeSSIM(firstSingleArray, secondSingleArray, width, height);
        }

        public class SSIM
        {
            private const double K1 = 0.01;
            private const double K2 = 0.03;
            private const double L = 255;
            private readonly Grid window = Gaussian(11, 1.5);

            public float ComputeSSIM(float[] img1, float[] img2, int width, int height)
            {
                var grid1 = new Grid(width, height);
                var grid2 = new Grid(width, height);
                Grid.Op((x, y) => img1[(y * width) + x], grid1);
                Grid.Op((x, y) => img2[(y * width) + x], grid2);

                return (float)this.ComputeSSIM(grid1, grid2);
            }

            private static Grid Gaussian(int size, double sigma)
            {
                var filter = new Grid(size, size);
                double s2 = sigma * sigma, c = (size - 1) / 2.0, dx, dy;

                Grid.Op(
                    (i, j) =>
                    {
                        dx = i - c;
                        dy = j - c;
                        return Math.Exp(-((dx * dx) + (dy * dy)) / (2 * s2));
                    },
                    filter);

                var scale = 1.0 / filter.Total;
                Grid.Op((i, j) => filter[i, j] * scale, filter);
                return filter;
            }

            private static Grid SubSample(Grid img, int skip)
            {
                int w = img.Width;
                int h = img.Height;
                double scale = 1.0 / (skip * skip);
                var ans = new Grid(w / skip, h / skip);
                for (int i = 0; i < w - skip; i += skip)
                {
                    for (int j = 0; j < h - skip; j += skip)
                    {
                        double sum = 0;
                        for (int x = i; x < i + skip; ++x)
                        {
                            for (int y = j; y < j + skip; ++y)
                            {
                                sum += img[x, y];
                            }
                        }

                        ans[i / skip, j / skip] = sum * scale;
                    }
                }

                return ans;
            }

            private static Grid Filter(Grid a, Grid b)
            {
                // todo - check and clean this                                                                                                          
                int ax = a.Width, ay = a.Height;
                int bx = b.Width, by = b.Height;
                int bcx = (bx + 1) / 2, bcy = (by + 1) / 2; // center position                                                                          
                var c = new Grid(ax - bx + 1, ay - by + 1);
                for (int i = bx - bcx + 1; i < ax - bx; ++i)
                {
                    for (int j = by - bcy + 1; j < ay - by; ++j)
                    {
                        double sum = 0;
                        for (int x = bcx - bx + 1 + i; x < 1 + i + bcx; ++x)
                        {
                            for (int y = bcy - by + 1 + j; y < 1 + j + bcy; ++y)
                            {
                                sum += a[x, y] * b[bx - bcx - 1 - i + x, by - bcy - 1 - j + y];
                            }
                        }

                        c[i - bcx, j - bcy] = sum;
                    }
                }

                return c;
            }

            private static Grid Linear(double s, Grid a, double c)
            {
                return Grid.Op((i, j) => (s * a[i, j]) + c, new Grid(a.Width, a.Height));
            }

            private double ComputeSSIM(Grid img1, Grid img2)
            {
                // uses notation from paper                                                                                                             
                // automatic downsampling                                                                                                               
                int f = (int)Math.Max(1, Math.Round(Math.Min(img1.Width, img1.Height) / 256.0));
                if (f > 1)
                { // downsampling by f                                                                                                              
                  // use a simple low-pass filter and subsample by f                                                                                
                    img1 = SubSample(img1, f);
                    img2 = SubSample(img2, f);
                }

                // normalize window - todo - do in window set {}                                                                                        
                double scale = 1.0 / window.Total;
                Grid.Op((i, j) => window[i, j] * scale, window);

                // image statistics                                                                                                                     
                var mu1 = Filter(img1, window);
                var mu2 = Filter(img2, window);

                var mu1mu2 = mu1 * mu2;
                var mu1SQ = mu1 * mu1;
                var mu2SQ = mu2 * mu2;

                var sigma12 = Filter(img1 * img2, window) - mu1mu2;
                var sigma1SQ = Filter(img1 * img1, window) - mu1SQ;
                var sigma2SQ = Filter(img2 * img2, window) - mu2SQ;

                // constants from the paper                                                                                                             
                var c1 = K1 * L;
                c1 *= c1;
                var c2 = K2 * L;
                c2 *= c2;

                Grid? ssim_map = null;
                if ((c1 > 0) && (c2 > 0))
                {
                    ssim_map = Grid.Op(
                        (i, j) =>
                        ((2 * mu1mu2[i, j]) + c1) * ((2 * sigma12[i, j]) + c2) /
                        (mu1SQ[i, j] + mu2SQ[i, j] + c1) / (sigma1SQ[i, j] + sigma2SQ[i, j] + c2),
                        new Grid(mu1mu2.Width, mu1mu2.Height));
                }
                else
                {
                    var num1 = Linear(2, mu1mu2, c1);
                    var num2 = Linear(2, sigma12, c2);
                    var den1 = Linear(1, mu1SQ + mu2SQ, c1);
                    var den2 = Linear(1, sigma1SQ + sigma2SQ, c2);

                    var den = den1 * den2; // total denominator                                                                                         
                    ssim_map = new Grid(mu1.Width, mu1.Height);
                    for (int i = 0; i < ssim_map.Width; ++i)
                    {
                        for (int j = 0; j < ssim_map.Height; ++j)
                        {
                            ssim_map[i, j] = 1;
                            if (den[i, j] > 0)
                            {
                                ssim_map[i, j] = num1[i, j] * num2[i, j] / (den1[i, j] * den2[i, j]);
                            }
                            else if ((den1[i, j] != 0) && (den2[i, j] == 0))
                            {
                                ssim_map[i, j] = num1[i, j] / den1[i, j];
                            }
                        }
                    }
                }

                // average all values                                                                                                                   
                return ssim_map.Total / (ssim_map.Width * ssim_map.Height);
            }

            private class Grid
            {
                private readonly double[,] data;

                public Grid(int w, int h)
                {
                    data = new double[w, h];
                    Width = w;
                    Height = h;
                }

                public int Width { get; set; }

                public int Height { get; set; }

                public double Total
                {
                    get
                    {
                        double s = 0;
                        foreach (var d in data)
                        {
                            s += d;
                        }

                        return s;
                    }
                }

                public double this[int i, int j]
                {
                    get { return data[i, j]; }
                    set { data[i, j] = value; }
                }

                public static Grid operator +(Grid a, Grid b)
                {
                    return Op((i, j) => a[i, j] + b[i, j], new Grid(a.Width, a.Height));
                }

                public static Grid operator -(Grid a, Grid b)
                {
                    return Op((i, j) => a[i, j] - b[i, j], new Grid(a.Width, a.Height));
                }

                public static Grid operator *(Grid a, Grid b)
                {
                    return Op((i, j) => a[i, j] * b[i, j], new Grid(a.Width, a.Height));
                }

                public static Grid operator /(Grid a, Grid b)
                {
                    return Op((i, j) => a[i, j] / b[i, j], new Grid(a.Width, a.Height));
                }

                public static Grid Op(Func<int, int, double> f, Grid g)
                {
                    int w = g.Width, h = g.Height;
                    for (int i = 0; i < w; ++i)
                    {
                        for (int j = 0; j < h; ++j)
                        {
                            g[i, j] = f(i, j);
                        }
                    }

                    return g;
                }
            }
        }
    }
}
