namespace NLMBaseGUI.NLM
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.IO.Pipes;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Newtonsoft.Json;
    using NLMShared.Dtos;
    using NLMShared.Pipes;

    public class ExternalSession
        : ISession
    {
        private RunConfigDto? runConfig;
        private Process? runnerProcess;

        public ExternalSession(int sigma, string inputPath, string libraryPath)
        {
            this.runConfig = new RunConfigDto
            {
                Sigma = sigma,
                InputPath = inputPath,
                LibraryPath = libraryPath,
            };
        }

        public async Task Run()
        {
            var configSerialized = JsonConvert.SerializeObject(this.runConfig, Formatting.None);
            this.runConfig = null;

            this.runnerProcess = new Process();
            this.runnerProcess.StartInfo.FileName = "NLMRunner.exe";
            this.runnerProcess.StartInfo.UseShellExecute = true;
            this.runnerProcess.Start();

            var serverPipe = new ServerPipe("testpipe", x => x.StartByteReaderAsync());
            serverPipe.DataReceived += (sndr, args) => this.runConfig = JsonConvert.DeserializeObject<RunConfigDto>(args.String);
            serverPipe.Connected += async (sndr, args) => 
            { 
                await serverPipe.WriteString(configSerialized);
            };

            await Task.Run(() => this.runnerProcess.WaitForExit());

            this.runnerProcess.Close();
        }

        public async Task Cancel()
        {
            if (this.runnerProcess != null)
            {
                await Task.Run(() => this.runnerProcess.Kill());
            }
        }
    }
}
