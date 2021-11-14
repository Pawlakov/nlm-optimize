namespace NLMBaseGUI.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
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
            var runConfig = this.PrepareConfig();
            var configSerialized = JsonConvert.SerializeObject(runConfig, Formatting.None);

            this.runnerProcess = new Process();
            this.runnerProcess.StartInfo.FileName = "NLMRunner.exe";
            this.runnerProcess.StartInfo.UseShellExecute = true;
            this.runnerProcess.Start();

            var runResult = (RunResultDto?)null;

            var serverPipe = new ServerPipe("testpipe", x => x.StartByteReaderAsync());
            serverPipe.DataReceived += (sndr, args) => runResult = JsonConvert.DeserializeObject<RunResultDto>(args.String);
            serverPipe.Connected += async (sndr, args) => await serverPipe.WriteString(configSerialized);

            await Task.Run(() => this.runnerProcess.WaitForExit());
            this.runnerProcess.Close();

            if (runResult != null)
            {
                if (runResult.Exception != null)
                {
                    throw runResult.Exception;
                }
                else
                {
                    var filteredFile = new FileInfo(runResult.OutputPath);
                    var filtered = new Bitmap(filteredFile.FullName);
                    filteredFile.Delete();

                    var stats = this.CalculateStats(raw, this.input, filtered, runResult.Time);

                    return (filtered, stats);
                }
            }
            else
            {
                throw new ApplicationException("Wystąpił niemożliwy do obsłużenia błąd krytyczny.");
            }
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
                InputPath = Path.GetTempFileName(),
            };

            var file = new FileInfo(config.InputPath);
            this.input.Save(file.FullName);

            return config;
        }
    }
}
