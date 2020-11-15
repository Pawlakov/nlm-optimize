namespace NLMBase
{
    using System;

    public class SSIM
    {
        internal double K1 = 0.01, K2 = 0.03;
        internal double L = 255;
        readonly Grid window = Gaussian(11, 1.5);

        public float ComputeSSIM(float[] img1, float[] img2, int width, int height)
        {
            var grid1 = new Grid(width, height);
            var grid2 = new Grid(width, height);
            Grid.Op((x, y) => img1[y * width + x], grid1);
            Grid.Op((x, y) => img2[y * width + x], grid2);

            return (float)this.ComputeSSIM(grid1, grid2);
        }

        private double ComputeSSIM(Grid img1, Grid img2)
        {
            // uses notation from paper                                                                                                             
            // automatic downsampling                                                                                                               
            int f = (int)Math.Max(1, Math.Round(Math.Min(img1.width, img1.height) / 256.0));
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
            double C1 = K1 * L; C1 *= C1;
            double C2 = K2 * L; C2 *= C2;

            Grid ssim_map = null;
            if ((C1 > 0) && (C2 > 0))
            {
                ssim_map = Grid.Op((i, j) =>
                    (2 * mu1mu2[i, j] + C1) * (2 * sigma12[i, j] + C2) /
                    (mu1SQ[i, j] + mu2SQ[i, j] + C1) / (sigma1SQ[i, j] + sigma2SQ[i, j] + C2),
                    new Grid(mu1mu2.width, mu1mu2.height));
            }
            else
            {
                var num1 = Linear(2, mu1mu2, C1);
                var num2 = Linear(2, sigma12, C2);
                var den1 = Linear(1, mu1SQ + mu2SQ, C1);
                var den2 = Linear(1, sigma1SQ + sigma2SQ, C2);

                var den = den1 * den2; // total denominator                                                                                         
                ssim_map = new Grid(mu1.width, mu1.height);
                for (int i = 0; i < ssim_map.width; ++i)
                    for (int j = 0; j < ssim_map.height; ++j)
                    {
                        ssim_map[i, j] = 1;
                        if (den[i, j] > 0)
                            ssim_map[i, j] = num1[i, j] * num2[i, j] / (den1[i, j] * den2[i, j]);
                        else if ((den1[i, j] != 0) && (den2[i, j] == 0))
                            ssim_map[i, j] = num1[i, j] / den1[i, j];
                    }
            }

            // average all values                                                                                                                   
            return ssim_map.Total / (ssim_map.width * ssim_map.height);
        }

        private class Grid
        {
            private readonly double[,] data;
            public int width, height;
            public Grid(int w, int h)
            {
                data = new double[w, h];
                width = w;
                height = h;
            }

            public double this[int i, int j]
            {
                get { return data[i, j]; }
                set { data[i, j] = value; }
            }

            public double Total
            {
                get
                {
                    double s = 0;
                    foreach (var d in data) s += d;
                    return s;
                }
            }

            static public Grid operator +(Grid a, Grid b)
            {
                return Op((i, j) => a[i, j] + b[i, j], new Grid(a.width, a.height));
            }

            static public Grid operator -(Grid a, Grid b)
            {
                return Op((i, j) => a[i, j] - b[i, j], new Grid(a.width, a.height));
            }

            static public Grid operator *(Grid a, Grid b)
            {
                return Op((i, j) => a[i, j] * b[i, j], new Grid(a.width, a.height));
            }

            static public Grid operator /(Grid a, Grid b)
            {
                return Op((i, j) => a[i, j] / b[i, j], new Grid(a.width, a.height));
            }

            static public Grid Op(Func<int, int, double> f, Grid g)
            {
                int w = g.width, h = g.height;
                for (int i = 0; i < w; ++i)
                    for (int j = 0; j < h; ++j)
                        g[i, j] = f(i, j);
                return g;
            }
        }

        private static Grid Gaussian(int size, double sigma)
        {
            var filter = new Grid(size, size);
            double s2 = sigma * sigma, c = (size - 1) / 2.0, dx, dy;

            Grid.Op((i, j) =>
            {
                dx = i - c;
                dy = j - c;
                return Math.Exp(-(dx * dx + dy * dy) / (2 * s2));
            },
                filter);
            var scale = 1.0 / filter.Total;
            Grid.Op((i, j) => filter[i, j] * scale, filter);
            return filter;
        }

        private static Grid SubSample(Grid img, int skip)
        {
            int w = img.width;
            int h = img.height;
            double scale = 1.0 / (skip * skip);
            var ans = new Grid(w / skip, h / skip);
            for (int i = 0; i < w - skip; i += skip)
                for (int j = 0; j < h - skip; j += skip)
                {
                    double sum = 0;
                    for (int x = i; x < i + skip; ++x)
                        for (int y = j; y < j + skip; ++y)
                            sum += img[x, y];
                    ans[i / skip, j / skip] = sum * scale;
                }
            return ans;
        }

        private static Grid Filter(Grid a, Grid b)
        {
            // todo - check and clean this                                                                                                          
            int ax = a.width, ay = a.height;
            int bx = b.width, by = b.height;
            int bcx = (bx + 1) / 2, bcy = (by + 1) / 2; // center position                                                                          
            var c = new Grid(ax - bx + 1, ay - by + 1);
            for (int i = bx - bcx + 1; i < ax - bx; ++i)
                for (int j = by - bcy + 1; j < ay - by; ++j)
                {
                    double sum = 0;
                    for (int x = bcx - bx + 1 + i; x < 1 + i + bcx; ++x)
                        for (int y = bcy - by + 1 + j; y < 1 + j + bcy; ++y)
                            sum += a[x, y] * b[bx - bcx - 1 - i + x, by - bcy - 1 - j + y];
                    c[i - bcx, j - bcy] = sum;
                }
            return c;
        }

        private static Grid Linear(double s, Grid a, double c)
        {
            return Grid.Op((i, j) => s * a[i, j] + c, new Grid(a.width, a.height));
        }
    }
}