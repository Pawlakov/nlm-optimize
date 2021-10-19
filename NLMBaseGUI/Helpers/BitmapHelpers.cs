namespace NLMBaseGUI.Helpers
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public static class BitmapHelpers
    {
        public static float[] UnwrapChannels(byte[] input, int channels, int width, int height, int stride)
        {
            var output = new float[channels * width * height];
            for (var i = 0; i < channels; ++i)
            {
                for (var j = 0; j < height; ++j)
                {
                    for (var k = 0; k < width; ++k)
                    {
                        output[(width * ((height * i) + j)) + k] = input[(j * stride) + (k * channels) + i];
                    }
                }
            }

            return output;
        }

        public static byte[] WrapChannels(float[] input, int channels, int width, int height, int length, int stride)
        {
            var output = new byte[length];
            for (var i = 0; i < channels; ++i)
            {
                for (var j = 0; j < height; ++j)
                {
                    for (var k = 0; k < width; ++k)
                    {
                        var value = input[(width * ((height * i) + j)) + k];
                        output[(j * stride) + (k * channels) + i] = (byte)Math.Clamp(Math.Floor(value + 0.5), 0.0, 255.0);
                    }
                }
            }

            return output;
        }

        public static float[] MakeEmptyChannels(int channels, int width, int height)
        {
            var resultChannels = new float[channels * width * height];

            return resultChannels;
        }

        public static void WriteBitemapTheDumbWay(Bitmap bitmap, byte[] bytesWrapped, int channels, int width, int height, int stride)
        {
            for (var x = 0; x < width; ++x)
            {
                for (var y = 0; y < height; ++y)
                {
                    switch (channels)
                    {
                        case 1:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 1) + 0], bytesWrapped[(y * stride) + (x * 1) + 0], (bytesWrapped[(y * stride) + x] * 1) + 0));
                            break;
                        case 2:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 2) + 1], bytesWrapped[(y * stride) + (x * 2) + 0], bytesWrapped[(y * stride) + (x * 2) + 0], bytesWrapped[(y * stride) + (x * 2) + 0]));
                            break;
                        case 3:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 3) + 2], bytesWrapped[(y * stride) + (x * 3) + 1], bytesWrapped[(y * stride) + (x * 3) + 0]));
                            break;
                        case 4:
                            bitmap.SetPixel(x, y, Color.FromArgb(bytesWrapped[(y * stride) + (x * 4) + 3], bytesWrapped[(y * stride) + (x * 4) + 2], bytesWrapped[(y * stride) + (x * 4) + 1], bytesWrapped[(y * stride) + (x * 4) + 0]));
                            break;
                    }
                }
            }
        }
    }
}
