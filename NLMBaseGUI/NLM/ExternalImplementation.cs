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

        public ExternalImplementation(FileInfo libraryFile)
        {
            this.Name = libraryFile.Name;

            this.libraryHandle = NativeLibrary.Load(libraryFile.FullName);
            var functionAddress = NativeLibrary.GetExport(this.libraryHandle, Symbol);
            this.Denoise = Marshal.GetDelegateForFunctionPointer(functionAddress, typeof(DenoiseFunction)) as DenoiseFunction;
        }

        public string Name { get; }

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
