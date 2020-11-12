namespace NLMBase
{
    using System;
    using System.CommandLine;
    using System.CommandLine.Invocation;
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
            var rootCommand = new RootCommand ("NLM")
            {
                inputOption,
                deviationOption,
            };

            var program = new Program();
            rootCommand.Handler = CommandHandler.Create<FileInfo, int>((i, s) =>
            {
                var fileName = i.FullName;
                program.Run(fileName, s);
            });

            await rootCommand.InvokeAsync(args);
        }

        public void Run(string inputName, int sigma)
        {
            using (var library = (IImplementation)ExternalImplementation.OpenImplementation("NLMBasic.dll") ?? new DefaultImplementation())
            {
                if (library == null)
                {
                    Console.WriteLine("Failed to open library.");
                }
                else
                {
                    var noisy = (Bitmap)null;
                    var output = (Bitmap)null;
                    var input = new Bitmap(inputName);
                    var timeStamp = string.Format("{0:yyyy-MM-dd_HH-mm-ss-fff}", DateTime.Now);

                    var denoiser = new Denoiser(input, library);
                    denoiser.Work(sigma, out noisy, out output);

                    noisy.Save($"noisy-{timeStamp}.png");
                    output.Save($"filtered-{timeStamp}.png");
                }
            }
        }
    }
}
