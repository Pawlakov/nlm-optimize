using ReactiveUI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NLMBaseGUI.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
    {
        private Bitmap rawImage;
        private Bitmap noisyImage;
        private Bitmap filteredImage;

        public MainWindowViewModel()
        {
            /*
            var input = new Bitmap(@"C:\Users\pmatu\Desktop\flasz.png");

            var width = Math.Min(input.Width, input.Width);
            var height = Math.Min(input.Height, input.Height);
            var inputData = input.LockBits(
                new Rectangle(0, 0, input.Width, input.Height),
                ImageLockMode.ReadOnly,
                input.PixelFormat);
            var stride = inputData.Stride;
            var pixelFormat = inputData.PixelFormat;
            var channels = Image.GetPixelFormatSize(pixelFormat) / 8;
            var inputOrigin = inputData.Scan0;
            var length = Math.Abs(inputData.Stride) * inputData.Height;
            this.rawImage = new byte[length];
            Marshal.Copy(inputOrigin, this.rawImage, 0, length);
            input.UnlockBits(inputData);
            */

            /*this.rawImage = File.ReadAllBytes(@"/home/pawlakov/Pulpit/ani uwu.jpg");*/

            this.ShowOpenFileDialog = new Interaction<Unit, string?>();
            this.ShowSaveFileDialog = new Interaction<Unit, string?>();

            this.LoadRawCommand = ReactiveCommand.Create(this.LoadRaw);
            this.MakeNoisyCommand = ReactiveCommand.Create(this.MakeNoisy);
            this.LoadNoisyCommand = ReactiveCommand.Create(this.LoadNoisy);
            this.SaveNoisyCommand = ReactiveCommand.Create(this.SaveNoisy);
            this.MakeFilteredCommand = ReactiveCommand.Create(this.MakeFiltered);
            this.SaveFilteredCommand = ReactiveCommand.Create(this.SaveFiltered);
        }

        public Bitmap RawImage
        {
            get => this.rawImage;
            set => this.RaiseAndSetIfChanged(ref this.rawImage, value);
        }

        public Bitmap NoisyImage
        {
            get => this.noisyImage;
            set => this.RaiseAndSetIfChanged(ref this.noisyImage, value);
        }

        public Bitmap FilteredImage
        {
            get => this.filteredImage;
            set => this.RaiseAndSetIfChanged(ref this.filteredImage, value);
        }

        public Interaction<Unit, string?> ShowOpenFileDialog { get; }

        public Interaction<Unit, string?> ShowSaveFileDialog { get; }

        public ReactiveCommand<Unit, Unit> LoadRawCommand { get; }

        public ReactiveCommand<Unit, Unit> MakeNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> LoadNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> SaveNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> MakeFilteredCommand { get; }

        public ReactiveCommand<Unit, Unit> SaveFilteredCommand { get; }

        private void LoadRaw()
        {
            this.ShowOpenFileDialog.Handle(Unit.Default).Subscribe(x => 
            {
                this.RawImage = new Bitmap(x);
            });
        }

        private void MakeNoisy()
        {
        }

        private void LoadNoisy()
        {
            this.ShowOpenFileDialog.Handle(Unit.Default).Subscribe(x =>
            {
                this.NoisyImage = new Bitmap(x);
            });
        }

        private void SaveNoisy()
        {
            this.ShowSaveFileDialog.Handle(Unit.Default).Subscribe(x =>
            {
                this.NoisyImage.Save(x);
            });
        }

        private void MakeFiltered()
        {
        }

        private void SaveFiltered()
        {
            this.ShowSaveFileDialog.Handle(Unit.Default).Subscribe(x =>
            {
                this.FilteredImage.Save(x);
            });
        }
    }
}
