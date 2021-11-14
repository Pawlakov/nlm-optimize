namespace NLMRunner.NLM
{
    using System;
    using System.IO;
    using System.Runtime.InteropServices;
    using NLMShared.NLM;

    public class ExternalImplementation
        : BaseImplementation
    {
        private const string Symbol = "_Z7DenoiseiiffPPfS0_iii";

        private IntPtr libraryHandle;
        private DenoiseFunction function;

        public ExternalImplementation(FileInfo libraryFile)
        {
            this.libraryHandle = NativeLibrary.Load(libraryFile.FullName);
            var functionAddress = NativeLibrary.GetExport(this.libraryHandle, Symbol);
            this.function = Marshal.GetDelegateForFunctionPointer(functionAddress, typeof(DenoiseFunction)) as DenoiseFunction;
        }

        public unsafe void RunDenoise(float[] inputArray, float[] outputArray, int sigma, int channels, int width, int height)
        {
            var nlmParams = this.MakeParams(sigma, channels);

            fixed (float* inputFlatPointer = &inputArray[0], outputFlatPointer = &outputArray[0])
            {
                var fpI = new float*[channels];
                var fpO = new float*[channels];
                for (int ii = 0; ii < channels; ii++)
                {
                    fpI[ii] = &inputFlatPointer[ii * width * height];
                    fpO[ii] = &outputFlatPointer[ii * width * height];
                }

                fixed (float** inputPointer = &fpI[0], outputPointer = &fpO[0])
                {
                    this.function(nlmParams.Win, nlmParams.Bloc, sigma, nlmParams.FiltPar, inputPointer, outputPointer, channels, width, height);
                }
            }
        }

        public override void Dispose()
        {
            if (this.libraryHandle != IntPtr.Zero)
            {
                NativeLibrary.Free(this.libraryHandle);
                this.libraryHandle = IntPtr.Zero;
            }
        }
    }
}