using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NLMBase
{
    public class ExternalImplementation : IImplementation
    {
        private IntPtr libraryHandle;

        private ExternalImplementation(string libraryName)
        {
            this.libraryHandle =  NativeLibrary.Load(libraryName);
            var functionAddress = NativeLibrary.GetExport(this.libraryHandle, "_Z7DenoiseiiffPPfS0_iii");
            this.Denoise = Marshal.GetDelegateForFunctionPointer(functionAddress, typeof(DenoiseFunction)) as DenoiseFunction;
        }

        public static IImplementation OpenImplementation(string libraryName)
        {
            try
            {
                return new ExternalImplementation(libraryName);
            }
            catch
            {
                return null;
            }
        }

        public DenoiseFunction Denoise { get; }

        public void Dispose()
        {
            if (this.libraryHandle != IntPtr.Zero)
            {
                NativeLibrary.Free(this.libraryHandle);
            }
        }
    }
}
