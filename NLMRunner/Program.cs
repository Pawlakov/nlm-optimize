namespace NLMRunner
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.IO.Pipes;
    using Newtonsoft.Json;
    using NLMShared.Dtos;
    using NLMShared.Pipes;

    public class Program
    {
        public static async System.Threading.Tasks.Task Main(string[] args)
        {
            var config = (RunConfigDto)null;

            var clientPipe = new ClientPipe(".", "testpipe", x => x.StartByteReaderAsync());
            clientPipe.DataReceived += (sndr, args) => 
            {
                config = JsonConvert.DeserializeObject<RunConfigDto>(args.String);
            };

            clientPipe.Connect();

            while (config == null) { }

            Console.WriteLine("Jest konfig. Naciśnij klawisz ty dzbanie.");
            Console.ReadKey();
            var configSerialized = JsonConvert.SerializeObject(config, Formatting.None);

            await clientPipe.WriteString(configSerialized);
        }
    }
}
