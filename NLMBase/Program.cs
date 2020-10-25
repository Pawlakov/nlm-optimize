namespace NLMBase
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;

    public class Program
    {
        private const string HighLibraryName = "HighLevel.dll";

        private static int h;

        public static void Run()
        {
            Console.WriteLine("Standard deviation?");
            h = int.Parse(GetInput(x => !int.TryParse(x, out var output)));

            var library = Implementation.OpenImplementation(HighLibraryName);
            if (library == null)
            {
                Console.WriteLine("Failed to open library.");
            }
            else
            {
                long elapsed;
                using (library)
                {
                    elapsed = RevealMessage(library);
                }
                Console.WriteLine("Completed in {0} ticks.", elapsed);
            }
        }

        public static long RevealMessage(Implementation library)
        {
            long elapsed;
            Bitmap output;
            Console.WriteLine("What is the name of the combined image (WITH extension)?");
            var combined = new Bitmap(GetInput(x => !Try(() => new Bitmap(Image.FromFile(x ?? string.Empty)))));
            Console.WriteLine("What should be the name of the revealed image (WITHOUT extension)?");
            var revealed = GetInput(x => x.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0);
            using (var decoder = new Denoiser(combined, library))
            {
                elapsed = decoder.Denoise(h, 1, out output);
            }

            output.Save($"{revealed}.png");
            return elapsed;
        }

        public static int GetBytesPerPixel(PixelFormat pixelFormat)
        {
            switch (pixelFormat)
            {
                case PixelFormat.Format24bppRgb:
                    return 3;
                case PixelFormat.Format32bppArgb:
                case PixelFormat.Format32bppPArgb:
                case PixelFormat.Format32bppRgb:
                    return 4;
                default:
                    throw new ArgumentException("Only 24 and 32 bit images are supported");
            }
        }

        private static string GetInput(Predicate<string> validation)
        {
            string input;
            bool error;
            do
            {
                Console.Write("> ");
                input = Console.ReadLine();
                error = validation(input);
                if (error)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                    Console.Write(new string(' ', Console.WindowWidth));
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                }
            }
            while (error);

            return input;
        }

        private static bool Try(Action action)
        {
            try
            {
                action();
            }
            catch
            {
                return false;
            }
            return true;
        }
    }
}
