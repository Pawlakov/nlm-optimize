namespace NLMBase
{
    using System;
    using System.CommandLine;
    using System.CommandLine.Invocation;
    using System.CommandLine.Parsing;
    using System.Drawing;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;

    public class Program
    {
        public static async Task Main(string[] args)
        {
            var inputOption = new Option<FileInfo>("-i", "Input file")
            {
                IsRequired = true,
            };
            inputOption.AddValidator(a => 
                a.Tokens
                     .Select(t => t.Value)
                     .Where(filePath => !Denoiser.CheckInputFile(filePath))
                     .Select(t => $"File {t} is not a valid image or does not exist.")
                     .FirstOrDefault());

            var deviationOption = new Option<int>("-s", "Sigma")
            {
                IsRequired = true,
            };
            deviationOption.AddValidator(a => 
                a.Tokens
                     .Select(t => t.Value)
                     .Where(value => !Denoiser.CheckSigma(value))
                     .Select(t => $"Value {t} is not an integer number between 1 and 100.")
                     .FirstOrDefault());

            var libraryOption = new Option<FileInfo>("-l", "Denoising library")
            {
                IsRequired = false,
            };
            libraryOption.AddValidator(a => 
                a.Tokens
                     .Select(t => t.Value)
                     .Where(filePath => !File.Exists(filePath))
                     .Select(t => $"File {t} does not exist.")
                     .FirstOrDefault());

            var rootCommand = new RootCommand ("NLM")
            {
                inputOption,
                deviationOption,
                libraryOption,
            };

            ValidateSymbol<CommandResult> validator = (symbolResult) => 
            {
                var options = symbolResult.Children.Select(x => x as OptionResult).Where(x => x != null);
                var sigmaOption = options.FirstOrDefault(x => x?.Symbol?.Name == "s");
                var inputOption = options.FirstOrDefault(x => x?.Symbol?.Name == "i");

                return null;
            };
            rootCommand.AddValidator(validator);

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
                try
                {
                    implementation = ExternalImplementation.OpenImplementation(library.FullName);
                }
                catch (Exception exception)
                {
                    Console.WriteLine("Failed to open the dynamic library. {0}", exception.Message);
                }
            }
            else
            {
                implementation = new DefaultImplementation();
            }

            if (implementation != null)
            {
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
}