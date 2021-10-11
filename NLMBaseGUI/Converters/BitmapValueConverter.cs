using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NLMBaseGUI.Converters
{
    public class BitmapValueConverter : IValueConverter
    {
        public static BitmapValueConverter Instance = new BitmapValueConverter();

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
            {
                return null;
            }

            if (value is System.Drawing.Bitmap input && targetType.IsAssignableFrom(typeof(Bitmap)))
            {
                using (var outerStream = new MemoryStream())
                {
                    input.Save(outerStream, ImageFormat.Png);
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
