namespace NLMBaseGUI.NLM
{
    using NLMBaseGUI.Models;
    using NLMShared.Models;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;

    public class InternalSession
        : BaseSession
    {
        private int sigma;
        private Bitmap input;
        private CancellationTokenSource? tokenSource;

        public InternalSession(int sigma, Bitmap input)
        {
            this.sigma = sigma;
            this.input = input;
        }

        public override async Task<(Bitmap, FilteringStatsModel)> Run(Bitmap? raw)
        {
            Bitmap output;
            var result = new FilteringStatsModel();

            this.tokenSource = new CancellationTokenSource();
            var token = tokenSource.Token;

            using (var implementation = new DefaultImplementation())
            {
                var inputModel = BitmapModel.Create(this.input);
                var outputModel = BitmapModel.Create(inputModel.Width, inputModel.Height, inputModel.PixelFormat);

                var watch = Stopwatch.StartNew();
                await Task.Run(() => implementation.RunDenoise(inputModel.Data, outputModel.Data, this.sigma, inputModel.Channels, inputModel.Width, inputModel.Height, token));
                watch.Stop();

                output = outputModel.ToBitmap();

                result.Time = TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds);
            }

            this.tokenSource.Dispose();
            this.tokenSource = null;

            return (output, result);
        }

        public override async Task Cancel()
        {
            if (this.tokenSource != null)
            {
                await Task.Run(() => this.tokenSource.Cancel());
            }
        }
    }
}
