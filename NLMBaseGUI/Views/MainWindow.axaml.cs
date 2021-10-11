using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.ReactiveUI;
using NLMBaseGUI.ViewModels;
using ReactiveUI;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;

namespace NLMBaseGUI.Views
{
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
                d(this.ViewModel.ShowOpenFileDialog.RegisterHandler(ShowOpenFileDialog));
                d(this.ViewModel.ShowSaveFileDialog.RegisterHandler(ShowSaveFileDialog));
            });
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }

        private async Task ShowOpenFileDialog(InteractionContext<Unit, string?> interaction)
        {
            var dialog = new OpenFileDialog();
            var fileNames = await dialog.ShowAsync(this);
            interaction.SetOutput(fileNames.FirstOrDefault());
        }

        private async Task ShowSaveFileDialog(InteractionContext<Unit, string?> interaction)
        {
            var dialog = new SaveFileDialog();
            var fileName = await dialog.ShowAsync(this);
            interaction.SetOutput(fileName);
        }
    }
}
