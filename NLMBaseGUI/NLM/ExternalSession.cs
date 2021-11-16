namespace NLMBaseGUI.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;
    using System.IO.Pipes;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Newtonsoft.Json;
    using NLMBaseGUI.Models;
    using NLMShared.Dtos;
    using NLMShared.Pipes;

    public class ExternalSession
        : BaseSession
    {
        private int sigma;
        private Bitmap input;
        private string? libraryPath;
        private Process? runnerProcess;

        public ExternalSession(int sigma, Bitmap input, string? libraryPath)
        {
            this.sigma = sigma;
            this.input = input;
            this.libraryPath = libraryPath;
        }

        public override async Task<(Bitmap, FilteringStatsModel)> Run(Bitmap? raw)
        {
            var filtered = (Bitmap)null;
            var stats = (FilteringStatsModel)null;

            var runConfig = this.PrepareConfig();
            var configSerialized = JsonConvert.SerializeObject(runConfig, Formatting.None);

            this.runnerProcess = new Process();
            this.runnerProcess.StartInfo.FileName = "NLMRunner.exe";
            this.runnerProcess.StartInfo.UseShellExecute = false;
            this.runnerProcess.Start();

            var runResult = (RunResultDto?)null;

            var serverPipe = new ServerPipe("testpipe", x => x.StartByteReaderAsync());
            serverPipe.Connected += async (sndr, args) => await serverPipe.WriteString(configSerialized);
            serverPipe.DataReceived += (sndr, args) =>
            {
                runResult = JsonConvert.DeserializeObject<RunResultDto>(args.String);
            };
            serverPipe.PipeClosed += (sndr, args) => 
            {
                this.runnerProcess.WaitForExit();
                this.runnerProcess.Close();

                if (runResult != null)
                {
                    if (runResult.Exception != null)
                    {
                        throw runResult.Exception;
                    }
                    else
                    {
                        using (var filteredFile = new MemoryStream(Convert.FromBase64String(runResult.OutputFile)))
                        {
                            filtered = new Bitmap(filteredFile);
                        }

                        stats = this.CalculateStats(raw, filtered, runResult.Time);
                    }
                }
                else
                {
                    throw new ApplicationException("Wystąpił niemożliwy do obsłużenia błąd krytyczny.");
                }
            };

            await Task.Run(() =>
            {
                while (filtered == null || stats == null)
                { }
            });

            return (filtered, stats);
        }

        public override async Task Cancel()
        {
            if (this.runnerProcess != null)
            {
                await Task.Run(() => this.runnerProcess.Kill());
            }
        }

        private RunConfigDto PrepareConfig()
        {
            var config = new RunConfigDto
            {
                Sigma = this.sigma,
                LibraryPath = this.libraryPath,
            };

            using (var inputFile = new MemoryStream())
            {
                this.input.Save(inputFile, ImageFormat.Png);
                config.InputFile = Convert.ToBase64String(inputFile.ToArray());
            }

            return config;
        }
    }
}
