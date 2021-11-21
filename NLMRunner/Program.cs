﻿namespace NLMRunner
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

            var clientPipe = new ClientPipe(".", "testpipe", x => x.StartByteReaderAsync());
            clientPipe.DataReceived += (sndr, args) =>
            {
                config = JsonConvert.DeserializeObject<RunConfigDto>(args.String);
                Console.WriteLine("Odebrano konfigurację");
            };

            clientPipe.Connect();

            while (config == null)
            {
            }

            try
            {
                var implementationFile = string.IsNullOrWhiteSpace(config.LibraryPath) ? null : new FileInfo(config.LibraryPath);
                using (var implementation = (IImplementation)(implementationFile != null ? new ExternalImplementation(implementationFile) : new DefaultImplementation()))
                {
                    var input = (Bitmap)null;
                    using (var inputFile = new MemoryStream(Convert.FromBase64String(config.InputFile)))
                    {
                        input = new Bitmap(inputFile);
                    }

                    var inputModel = BitmapModel.Create(input);
                    var outputModel = BitmapModel.Create(inputModel.Width, inputModel.Height, inputModel.PixelFormat);

                    var watch = Stopwatch.StartNew();
                    implementation.RunDenoise(inputModel.Data, outputModel.Data, config.Sigma, inputModel.Channels, inputModel.Width, inputModel.Height);
                    watch.Stop();

                    var output = outputModel.ToBitmap();
                    using (var outputFile = new MemoryStream())
                    {
                        output.Save(outputFile, ImageFormat.Png);
                        result.OutputFile = Convert.ToBase64String(outputFile.ToArray());
                    }

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
                Console.WriteLine("Odesłano wyniki");
            }
        }
    }
}
