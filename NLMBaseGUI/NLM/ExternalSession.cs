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
    using NLMShared.Models;
    using NLMShared.Pipes;

    public class ExternalSession
        : BaseSession
    {
        private int sigma;
        private int windowRadius;
        private int blockRadius;
        private float filterParam;
        private Bitmap input;
        private string libraryPath;
        private Process runnerProcess;
        private bool cancelled;

        public ExternalSession(int sigma, int windowRadius, int blockRadius, float filterParam, Bitmap input, string libraryPath)
        {
            this.sigma = sigma;
            this.windowRadius = windowRadius;
            this.blockRadius = blockRadius;
            this.filterParam = filterParam;
            this.input = input;
            this.libraryPath = libraryPath;
        }

        public override async Task<(Bitmap, FilteringStatsModel)> Run(Bitmap raw)
        {
            var filtered = (Bitmap)null;
            var stats = (FilteringStatsModel)null;
            var runnerExited = false;
            var runnerException = (Exception)null;

            var runConfig = this.PrepareConfig();

            var runResult = (RunResultDto)null;

            var serverPipe = new ServerPipe("testpipe", x => x.StartObjectReaderAsync());
            serverPipe.Connected += async (sndr, args) =>
            {
                try
                {
                    serverPipe.Flush();
                    await serverPipe.WriteObject(runConfig);
                    serverPipe.Flush();
                }
                catch (Exception exception)
                {
                    runnerException = exception;
                }
            };
            serverPipe.DataReceived += (sndr, args) =>
            {
                if (args.Error == null)
                {
                    runResult = (RunResultDto)args.ObjectData;
                }
                else
                {
                    runnerException = args.Error;
                }
            };

            this.runnerProcess = Process.Start(new ProcessStartInfo
            {
#if Linux
                FileName = "dotnet",
                Arguments = Path.Combine("bin", "Debug", "net5.0", "NLMRunner.dll"),
                ErrorDialog = true,
#elif Windows
                FileName = "NLMRunner.exe",
#endif
            });

            await this.runnerProcess.WaitForExitAsync();

            if (runResult != null)
            {
                if (runResult.Exception != null)
                {
                    runnerException = runResult.Exception;
                }
                else
                {
                    using (var filteredFile = new MemoryStream(runResult.OutputFile))
                    {
                        filtered = new Bitmap(filteredFile);
                    }

                    stats = this.CalculateStats(raw, filtered, runResult.Time);
                }
            }
            else if (!this.cancelled)
            {
                runnerException = new ApplicationException("Wystąpił niemożliwy do obsłużenia błąd krytyczny.");
            }

            runnerExited = true;

            if (runnerException != null)
            {
                throw runnerException;
            }

            if (filtered == null || stats == null)
            {
                throw new ApplicationException("Proces zakończył się bez rezultatu.");
            }

            return (filtered, stats);
        }

        public override async Task Cancel()
        {
            if (this.runnerProcess != null)
            {
                this.cancelled = true;
                await Task.Run(() => this.runnerProcess.Kill());
            }
        }

        private RunConfigDto PrepareConfig()
        {
            var config = new RunConfigDto
            {
                LibraryPath = this.libraryPath,
                Params = new NLMParamsModel
                {
                    Sigma = this.sigma,
                    Bloc = this.blockRadius,
                    FiltPar = this.filterParam,
                    Win = this.windowRadius,
                },
            };

            using (var inputFile = new MemoryStream())
            {
                this.input.Save(inputFile, ImageFormat.Png);
                config.InputFile = inputFile.ToArray();
            }

            return config;
        }
    }
}
