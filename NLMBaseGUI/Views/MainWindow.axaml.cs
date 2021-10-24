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
    using global::MessageBox.Avalonia;
    using global::MessageBox.Avalonia.DTO;
    using global::MessageBox.Avalonia.Enums;
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
                d(this.ViewModel!.ShowOpenLibraryDialog.RegisterHandler(this.ShowOpenLibraryDialog));
                d(this.ViewModel!.ShowOpenImageDialog.RegisterHandler(this.ShowOpenImageDialog));
                d(this.ViewModel!.ShowSaveImageDialog.RegisterHandler(this.ShowSaveImageDialog));
                d(this.ViewModel!.ShowMessageBox.RegisterHandler(this.ShowMessageBox));
            });
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private async Task ShowOpenLibraryDialog(InteractionContext<Unit, string[]?> interaction)
        {
            var dialog = new OpenFileDialog
            {
                AllowMultiple = true,
                Filters = new List<FileDialogFilter>
                {
                    new FileDialogFilter
                    {
                        Name = "Biblioteki",
                        Extensions = new List<string>
                        {
                            "dll",
                            "so",
                        },
                    },
                },
            };

            var fileNames = await dialog.ShowAsync(this);
            interaction.SetOutput(fileNames);
        }

        private async Task ShowOpenImageDialog(InteractionContext<Unit, string?> interaction)
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

        private async Task ShowSaveImageDialog(InteractionContext<Unit, string?> interaction)
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
            var msBoxStandardWindow = MessageBoxManager
                .GetMessageBoxStandardWindow(new MessageBoxStandardParams
                {
                    ButtonDefinitions = ButtonEnum.Ok,
                    ContentTitle = "B³¹d",
                    ContentMessage = interaction.Input,
                    Icon = global::MessageBox.Avalonia.Enums.Icon.Error,
                });

            await msBoxStandardWindow.Show();
        }
    }
}
