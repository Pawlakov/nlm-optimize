namespace NLMBaseGUI.NLM
{
    using System;
    using System.IO;
    using System.Runtime.InteropServices;

    public class ExternalImplementation 
        : IImplementation
    {
        private const string Symbol = "_Z7DenoiseiiffPPfS0_iii";

        private IntPtr libraryHandle;

        private ExternalImplementation(FileInfo libraryFile)
        {
            this.libraryHandle = NativeLibrary.Load(libraryFile.FullName);
            var functionAddress = NativeLibrary.GetExport(this.libraryHandle, Symbol);
            this.Denoise = Marshal.GetDelegateForFunctionPointer(functionAddress, typeof(DenoiseFunction)) as DenoiseFunction;
        }

        public static IImplementation OpenImplementation(string libraryName)
        {
            var file = new FileInfo(libraryName);
            if (!file.Exists)
            {
                throw new Exception("File not found.");
            }

            return new ExternalImplementation(file);
        }

        public DenoiseFunction Denoise { get; }

        public void Dispose()
        {
            if (this.libraryHandle != IntPtr.Zero)
            {
                NativeLibrary.Free(this.libraryHandle);
                this.libraryHandle = IntPtr.Zero;
            }
        }
    }
}
