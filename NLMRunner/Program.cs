namespace NLMRunner
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;
    using System.Threading.Tasks;
    using Newtonsoft.Json;
    using NLMRunner.NLM;
    using NLMShared.Dtos;
    using NLMShared.Models;
    using NLMShared.Pipes;

    public class Program
    {
        public static async Task Main(string[] args)
        {
            var config = (RunConfigDto)null;
            var result = new RunResultDto();

            var clientPipe = new ClientPipe(".", "testpipe", x => x.StartObjectReaderAsync());
            clientPipe.DataReceived += (sndr, args) =>
            {
                config = (RunConfigDto)args.ObjectData;
                Console.WriteLine("Odebrałęm konfigurację");
            };

            Console.WriteLine("Próbuję połączyć się z serwerem...");
            clientPipe.Connect();
            Console.WriteLine("Połączyłem się");

            while (config == null)
            {
            }

            try
            {
                var implementationFile = string.IsNullOrWhiteSpace(config.LibraryPath) ? null : new FileInfo(config.LibraryPath);
                using (var implementation = (IImplementation)(implementationFile != null ? new ExternalImplementation(implementationFile) : new DefaultImplementation()))
                {
                    var input = (Bitmap)null;
                    using (var inputFile = new MemoryStream(config.InputFile))
                    {
                        input = new Bitmap(inputFile);
                    }

                    var inputModel = BitmapModel.Create(input);
                    var outputModel = BitmapModel.Create(inputModel.Width, inputModel.Height, inputModel.PixelFormat);

                    Console.WriteLine("Rozpoczynam działanie filtra...");
                    var watch = Stopwatch.StartNew();
                    implementation.RunDenoise(inputModel.Data, outputModel.Data, config.Sigma, inputModel.Channels, inputModel.Width, inputModel.Height);
                    watch.Stop();
                    Console.WriteLine($"Przefiltrowałem ({TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds)})");

                    var output = outputModel.ToBitmap();
                    using (var outputFile = new MemoryStream())
                    {
                        output.Save(outputFile, ImageFormat.Png);
                        result.OutputFile = outputFile.ToArray();
                    }

                    result.Time = watch.ElapsedMilliseconds;
                }
            }
            catch (Exception exception)
            {
                Console.WriteLine("Złapałem wyjątek");
                result.Exception = exception;
            }
            finally
            {
                Console.WriteLine("Odsyłam wyniki...");
                var watch = Stopwatch.StartNew();
                clientPipe.Flush();
                await clientPipe.WriteObject(result);
                clientPipe.Flush();
                watch.Stop();
                Console.WriteLine($"Odesłałem ({TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds)})");
            }
        }
    }
}
