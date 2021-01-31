namespace NLMBase
{
    using System;
    using System.CommandLine;
    using System.CommandLine.Invocation;
    using System.CommandLine.Parsing;
    using System.Drawing;
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
            var deviationOption = new Option<int>("-s", "Sigma")
            {
                IsRequired = true,
            };
            var libraryOption = new Option<FileInfo>("-l", "Denoising library")
            {
                IsRequired = false,
            }.ExistingOnly();
            var rootCommand = new RootCommand ("NLM")
            {
                inputOption,
                deviationOption,
                libraryOption,
            };

            var program = new Program();
            rootCommand.Handler = CommandHandler.Create<FileInfo, int, FileInfo>((i, s, l) => 
            { 
                program.Run(i, s, l); 
            });

            await rootCommand.InvokeAsync(args);
        }

        public void Run(FileInfo input, int sigma, FileInfo library)
        {
            var implementation = (IImplementation)null;

            if (library != null)
            {
                implementation = ExternalImplementation.OpenImplementation(library.FullName);
                if (implementation == null)
                {
                    Console.WriteLine("Failed to open dynamic library.");
                    throw new Exception("Kupa. Zrobić to jak należy w dekalarcji polecenia cli.");
                }
            }

            if (implementation == null)
            {
                implementation = new DefaultImplementation();
            }

            var noisy = (Bitmap)null;
            var output = (Bitmap)null;
            var mseNoisy = 0.0f;
            var mseOutput = 0.0f;
            var ssimNoisy = 0.0f;
            var ssimOutput = 0.0f;
            var inputBitmap = new Bitmap(input.FullName);
            var timeStamp = string.Format("{0:yyyy-MM-dd_HH-mm-ss-fff}", DateTime.Now);

            var denoiser = new Denoiser(inputBitmap, implementation);
            var millisecondsElapsed = denoiser.Work(sigma, out noisy, out output, out mseNoisy, out mseOutput, out ssimNoisy, out ssimOutput);
            var time = TimeSpan.FromMilliseconds(millisecondsElapsed);
            Console.WriteLine("Time elapsed: {0}", time);
            Console.WriteLine("MSE: {0} -> {1}", mseNoisy, mseOutput);
            Console.WriteLine("SSIM: {0} -> {1}", ssimNoisy, ssimOutput);

            noisy.Save($"noisy-{timeStamp}.png");
            output.Save($"filtered-{timeStamp}.png");

            implementation.Dispose();
        }
    }
}
