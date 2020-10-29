namespace NLMBase
{
    using System;
    using System.Collections.Generic;
    using System.CommandLine;
    using System.CommandLine.Invocation;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;
    using System.Threading.Tasks;

    public class Program
    {
        public static async Task Main(string[] args)
        {
            var inputOption = new Option<FileInfo>("-i", "Input file")
            {
                IsRequired = true,
            }.ExistingOnly();
            var deviationOption = new Option<int>("-d", "Standard deviation")
            {
                IsRequired = true,
            };
            var rootCommand = new RootCommand ("NLM")
            {
                inputOption,
                deviationOption,
            };

            var program = new Program();
            rootCommand.Handler = CommandHandler.Create<FileInfo, int>((i, d) =>
            {
                var fileName = i.FullName;
                program.Run(fileName, d);
            });

            await rootCommand.InvokeAsync(args);
        }

        public void Run(string input, int h)
        {
            var library = new Implementation();
            if (library == null)
            {
                Console.WriteLine("Failed to open library.");
            }
            else
            {
                long elapsed;
                Bitmap output;
                Console.WriteLine(input);
                var combined = new Bitmap(input);
                var revealed = string.Format("{0:yyyy-MM-dd_HH-mm-ss-fff}", DateTime.Now);
                using (var decoder = new Denoiser(combined, library))
                {
                    elapsed = decoder.Denoise(h, out output);
                }

                output.Save($"{revealed}.png");
                Console.WriteLine("Completed in {0} ticks.", elapsed);
            }
        }

        public static int GetBytesPerPixel(PixelFormat pixelFormat)
        {
            switch (pixelFormat)
            {
                case PixelFormat.Format24bppRgb:
                    return 3;
                case PixelFormat.Format32bppArgb:
                case PixelFormat.Format32bppPArgb:
                case PixelFormat.Format32bppRgb:
                case PixelFormat.Format4bppIndexed:
                    return 4;
                default:
                    Console.WriteLine("{0}", pixelFormat);
                    throw new ArgumentException("Only 24 and 32 bit images are supported");
            }
        }
    }
}
