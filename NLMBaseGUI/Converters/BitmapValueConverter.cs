namespace NLMBaseGUI.Converters
{
    using System;
    using System.Drawing.Imaging;
    using System.Globalization;
    using System.IO;
    using Avalonia.Data.Converters;
    using Avalonia.Media.Imaging;
    using SkiaSharp;

    public class BitmapValueConverter
        : IValueConverter
    {
        public static BitmapValueConverter Instance { get; } = new BitmapValueConverter();

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
            {
                return null;
            }

            if (value is SKBitmap input && targetType.IsAssignableFrom(typeof(Bitmap)))
            {
                using (var outerStream = new MemoryStream())
                {
                    input.Encode(SKEncodedImageFormat.Png, 100).SaveTo(outerStream);
                    using (var innerStream = new MemoryStream(outerStream.ToArray()))
                    {
                        return new Bitmap(innerStream);
                    }
                }
            }

            throw new NotSupportedException();
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }
}
