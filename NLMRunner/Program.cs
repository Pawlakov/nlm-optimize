namespace NLMRunner
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
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

            var clientPipe = new ClientPipe(".", "testpipe", x => x.StartByteReaderAsync());
            clientPipe.DataReceived += (sndr, args) => 
            {
                config = JsonConvert.DeserializeObject<RunConfigDto>(args.String);
            };

            clientPipe.Connect();

            while (config == null) { }

            try
            {
                var implementationFile = new FileInfo(config.LibraryPath);
                using (var implementation = new ExternalImplementation(implementationFile))
                {
                    var inputFile = new FileInfo(config.InputPath);
                    var input = new Bitmap(inputFile.FullName);
                    var inputModel = BitmapModel.Create(input);
                    var outputModel = BitmapModel.Create(inputModel.Width, inputModel.Height, inputModel.PixelFormat);
                    inputFile.Delete();

                    var watch = Stopwatch.StartNew();
                    implementation.RunDenoise(inputModel.Data, outputModel.Data, config.Sigma, inputModel.Channels, inputModel.Width, inputModel.Height);
                    watch.Stop();

                    var output = outputModel.ToBitmap();
                    var outputFile = new FileInfo(Path.GetTempFileName());
                    output.Save(outputFile.FullName);

                    result.OutputPath = outputFile.FullName;
                    result.Time = watch.ElapsedMilliseconds;
                }
            }
            catch (Exception exception)
            {
                result.Exception = exception;
            }
            finally
            {
                var resultSerialized = JsonConvert.SerializeObject(result, Formatting.None);
                await clientPipe.WriteString(resultSerialized);
            }
        }
    }
}
