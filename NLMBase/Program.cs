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
                IsRequired = false,
            }.ExistingOnly();
            var inputDirOption = new Option<DirectoryInfo>("-d", "Input directory")
            {
                IsRequired = false,
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
                inputDirOption,
                deviationOption,
                libraryOption,
            };

            var program = new Program();
            rootCommand.Handler = CommandHandler.Create<FileInfo, DirectoryInfo, int, FileInfo>((i, d, s, l) =>
            {
                var directoryName = d?.FullName;
                var fileName = i?.FullName;
                var libraryName = l?.FullName;
                program.Run(fileName, directoryName, s, libraryName);
            });

            await rootCommand.InvokeAsync(args);
        }

        public void Run(string inputName, string directoryName, int sigma, string libraryName)
        {
            var library = (IImplementation)null;

            if (libraryName != null)
            {
                library = ExternalImplementation.OpenImplementation(libraryName);
                if (library == null)
                {
                    Console.WriteLine("Failed to open dynamic library. Using default implementation.");
                }
            }

            if (library == null)
            {
                library = new DefaultImplementation();
            }

            var noisy = (Bitmap)null;
            var output = (Bitmap)null;
            var mseNoisy = 0.0f;
            var mseOutput = 0.0f;
            var ssimNoisy = 0.0f;
            var ssimOutput = 0.0f;
            var input = new Bitmap(inputName);
            var timeStamp = string.Format("{0:yyyy-MM-dd_HH-mm-ss-fff}", DateTime.Now);

            var denoiser = new Denoiser(input, library);
            var millisecondsElapsed = denoiser.Work(sigma, out noisy, out output, out mseNoisy, out mseOutput, out ssimNoisy, out ssimOutput);
            var time = TimeSpan.FromMilliseconds(millisecondsElapsed);
            Console.WriteLine("Time elapsed: {0}", time);
            Console.WriteLine("MSE: {0} -> {1}", mseNoisy, mseOutput);
            Console.WriteLine("SSIM: {0} -> {1}", ssimNoisy, ssimOutput);

            noisy.Save($"noisy-{timeStamp}.png");
            output.Save($"filtered-{timeStamp}.png");

            library.Dispose();
        }
    }
}
