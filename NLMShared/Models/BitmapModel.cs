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

    public class BitmapModel
    {
        private BitmapModel()
        {
        }

        public int Width { get; private set; }

        public int Height { get; private set; }

        public int Channels { get; private set; }

        public int Stride { get; private set; }

        public PixelFormat PixelFormat { get; private set; }

        public int Length { get; private set; }

        public float[] Data { get; private set; }

        public static BitmapModel Create(Bitmap bitmap)
        {
            var model = new BitmapModel();

            model.Width = Math.Min(bitmap.Width, bitmap.Width);
            model.Height = Math.Min(bitmap.Height, bitmap.Height);
            var data = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadOnly,
                bitmap.PixelFormat);
            model.Stride = data.Stride;
            model.PixelFormat = data.PixelFormat;
            model.Channels = Image.GetPixelFormatSize(model.PixelFormat) / 8;
            var origin = data.Scan0;
            model.Length = data.Stride * data.Height;
            var array = new byte[model.Length];
            Marshal.Copy(origin, array, 0, model.Length);
            bitmap.UnlockBits(data);

            model.Data = BitmapHelpers.UnwrapChannels(array, model.Channels, model.Width, model.Height, model.Stride);

            return model;
        }

        public static BitmapModel Create(int width, int height, PixelFormat pixelFormat)
        {
            var model = new BitmapModel();

            model.Width = width;
            model.Height = height;
            model.PixelFormat = pixelFormat;
            model.Channels = Image.GetPixelFormatSize(pixelFormat) / 8;
            model.Data = new float[model.Channels * width * height];
            model.Stride = (int)(Math.Ceiling(width * model.Channels * 0.25) * 4);
            model.Length = model.Stride * model.Height;

            return model;
        }

        public Bitmap ToBitmap()
        {
            var bitmap = new Bitmap(this.Width, this.Height, this.PixelFormat);

            var array = BitmapHelpers.WrapChannels(this.Data, this.Channels, this.Width, this.Height, this.Length, this.Stride);
            BitmapHelpers.WriteBitemapTheDumbWay(bitmap, array, this.Channels, this.Width, this.Height, this.Stride);

            return bitmap;
        }
    }
}
