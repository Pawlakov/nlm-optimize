namespace NLMShared.Models
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Threading.Tasks;
    using NLMShared.Helpers;
    using SkiaSharp;

    public class BitmapModel
    {
        private BitmapModel()
        {
        }

        public int Width { get; private set; }

        public int Height { get; private set; }

        public int Channels { get; private set; }

        public int Stride { get; private set; }

        public SKColorType ColorType { get; private set; }

        public SKAlphaType AlphaType { get; private set; }

        public int Length { get; private set; }

        public float[] Data { get; private set; }

        public static BitmapModel Create(SKBitmap bitmap)
        {
            var model = new BitmapModel();

            model.Width = bitmap.Width;
            model.Height = bitmap.Height;
            model.ColorType = bitmap.ColorType;
            model.AlphaType = bitmap.AlphaType;
            model.Channels = model.ColorType.GetBytesPerPixel();
            model.Length = bitmap.ByteCount;
            model.Stride = bitmap.ByteCount / bitmap.Height;
            var origin = bitmap.GetPixels();
            var array = new byte[model.Length];
            Marshal.Copy(origin, array, 0, model.Length);

            model.Data = BitmapHelpers.UnwrapChannels(array, model.Channels, model.Width, model.Height, model.Stride);

            return model;
        }

        public static BitmapModel Create(int width, int height, SKColorType colorType, SKAlphaType alphaType)
        {
            var model = new BitmapModel();

            model.Width = width;
            model.Height = height;
            model.ColorType = colorType;
            model.AlphaType = alphaType;
            model.Channels = model.ColorType.GetBytesPerPixel();
            model.Data = new float[model.Channels * width * height];
            model.Stride = (int)(Math.Ceiling(width * model.Channels * 0.25) * 4);
            model.Length = model.Stride * model.Height;

            return model;
        }

        public SKBitmap ToBitmap()
        {
            var bitmap = new SKBitmap(this.Width, this.Height, this.ColorType, this.AlphaType);
            var origin = bitmap.GetPixels();
            var array = BitmapHelpers.WrapChannels(this.Data, this.Channels, this.Width, this.Height, this.Length, this.Stride);
            Marshal.Copy(array, 0, origin, this.Length);

            return bitmap;
        }
    }
}
