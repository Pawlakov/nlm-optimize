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
            this.libraryHandle = LoadLibrary(libraryName);
            if (this.libraryHandle == IntPtr.Zero)
            {
                throw new Exception(Marshal.GetLastWin32Error().ToString());
            }

            var functionAddress = GetProcAddress(this.libraryHandle, "Denoise");
            if (functionAddress == IntPtr.Zero)
            {
                throw new Exception(Marshal.GetLastWin32Error().ToString());
            }

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
                FreeLibrary(this.libraryHandle);
            }
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern IntPtr LoadLibrary(string libname);

        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool FreeLibrary(IntPtr hModule);

        [DllImport("kernel32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
        private static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);
    }
}
