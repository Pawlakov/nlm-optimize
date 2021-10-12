namespace NLMBaseGUI.Views
{
    using System.Collections.Generic;
    using System.Linq;
    using System.Reactive;
    using System.Threading.Tasks;
    using Avalonia;
    using Avalonia.Controls;
    using Avalonia.Markup.Xaml;
    using Avalonia.ReactiveUI;
    using NLMBaseGUI.ViewModels;
    using ReactiveUI;

    public partial class MainWindow : ReactiveWindow<MainWindowViewModel>
    {
        public MainWindow()
        {
            InitializeComponent();
#if DEBUG
            this.AttachDevTools();
#endif
            this.WhenActivated(d =>
            {
                d(this.ViewModel.ShowOpenFileDialog.RegisterHandler(this.ShowOpenFileDialog));
                d(this.ViewModel.ShowSaveFileDialog.RegisterHandler(this.ShowSaveFileDialog));
                d(this.ViewModel.ShowMessageBox.RegisterHandler(this.ShowMessageBox));
            });
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private async Task ShowOpenFileDialog(InteractionContext<Unit, string?> interaction)
        {
            var dialog = new OpenFileDialog
            {
                AllowMultiple = false,
                Filters = new List<FileDialogFilter>
                {
                    new FileDialogFilter
                    {
                        Name = "Obrazy",
                        Extensions = new List<string>
                        {
                            "bmp",
                            "png",
                            "jpg",
                            "jpeg",
                        },
                    },
                },
            };

            var fileNames = await dialog.ShowAsync(this);
            interaction.SetOutput(fileNames.FirstOrDefault());
        }

        private async Task ShowSaveFileDialog(InteractionContext<Unit, string?> interaction)
        {
            var dialog = new SaveFileDialog
            {
                DefaultExtension = "png",
                Filters = new List<FileDialogFilter>
                {
                    new FileDialogFilter
                    {
                        Name = "Obrazy",
                        Extensions = new List<string>
                        {
                            "bmp",
                            "png",
                            "jpg",
                            "jpeg",
                        },
                    },
                },
            };

            var fileName = await dialog.ShowAsync(this);
            interaction.SetOutput(fileName);
        }

        private async Task ShowMessageBox(InteractionContext<string, Unit> interaction)
        {
            await MessageBox.Show(this, interaction.Input, "Test title", MessageBox.MessageBoxButtons.Ok);
        }
    }
}
