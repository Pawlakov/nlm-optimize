namespace NLMBase
{
    public unsafe class Implementation
    {
        public void Denoise(byte* inputPointer, byte* outputPointer, int length, double h)
        {
            
        }

        private double CalculateC(int pixelIndex, int length)
        {
            var sum = 0.0;
            for (var i = 0; i < length; ++i)
            {
                sum += CalculateF(pixelIndex, i, length);
            }

            return sum;
        }

        private double CalculateF(int pixelIndex, int otherPixelIndex, int length)
        {

        }

        private double CalculateB(int pixelIndex)
        {

        }

        private double CalculateR()
        {

        }
    }
}