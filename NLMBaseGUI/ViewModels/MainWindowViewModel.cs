namespace NLMBaseGUI.ViewModels
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;
    using System.Reactive;
    using System.Reactive.Linq;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Threading.Tasks;
    using Avalonia.Threading;
    using NLMBaseGUI.Models;
    using NLMBaseGUI.Services;
    using ReactiveUI;

    public class MainWindowViewModel : ViewModelBase
    {
        private readonly NoiseService noiseService;
        private readonly FilterService filterService;

        private bool isProcessing;
        private int selectedTab;
        private Bitmap? rawImage;
        private Bitmap? noisyImage;
        private Bitmap? filteredImage;
        private int sigma;
        private FilteringStatsModel? filteringStats;

        public MainWindowViewModel()
        {
            this.noiseService = new NoiseService();
            this.filterService = new FilterService();

            this.ShowOpenFileDialog = new Interaction<Unit, string?>();
            this.ShowSaveFileDialog = new Interaction<Unit, string?>();
            this.ShowMessageBox = new Interaction<string, Unit>();

            this.LoadRawCommand = ReactiveCommand.Create(this.LoadRaw);
            this.MakeNoisyCommand = ReactiveCommand.CreateFromTask(this.MakeNoisy, this.WhenAnyValue(x => x.RawImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.LoadNoisyCommand = ReactiveCommand.Create(this.LoadNoisy);
            this.SaveNoisyCommand = ReactiveCommand.Create(this.SaveNoisy, this.WhenAnyValue(x => x.NoisyImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.MakeFilteredCommand = ReactiveCommand.CreateFromTask(this.MakeFiltered, this.WhenAnyValue(x => x.NoisyImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.SaveFilteredCommand = ReactiveCommand.Create(this.SaveFiltered, this.WhenAnyValue(x => x.FilteredImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
        }

        public bool IsProcessing
        {
            get => this.isProcessing;
            set => this.RaiseAndSetIfChanged(ref this.isProcessing, value);
        }

        public int SelectedTab
        {
            get => this.selectedTab;
            set => this.RaiseAndSetIfChanged(ref this.selectedTab, value);
        }

        public Bitmap? RawImage
        {
            get => this.rawImage;
            set => this.RaiseAndSetIfChanged(ref this.rawImage, value);
        }

        public Bitmap? NoisyImage
        {
            get => this.noisyImage;
            set => this.RaiseAndSetIfChanged(ref this.noisyImage, value);
        }

        public Bitmap? FilteredImage
        {
            get => this.filteredImage;
            set => this.RaiseAndSetIfChanged(ref this.filteredImage, value);
        }

        public string Sigma
        {
            get => this.sigma.ToString();
            set
            {
                if (int.TryParse(value, out var parsed))
                {
                    if (parsed > 100)
                    {
                        parsed = 100;
                    }
                    else if (parsed < 0)
                    {
                        parsed = 0;
                    }

                    this.sigma = parsed;
                }

                Debug.WriteLine(this.sigma);
                this.RaisePropertyChanged();
            }
        }

        public FilteringStatsModel? FilteringStats 
        {
            get => this.filteringStats;
            set => this.RaiseAndSetIfChanged(ref this.filteringStats, value);
        }

        public Interaction<Unit, string?> ShowOpenFileDialog { get; }

        public Interaction<Unit, string?> ShowSaveFileDialog { get; }

        public Interaction<string, Unit> ShowMessageBox { get; }

        public ReactiveCommand<Unit, Unit> LoadRawCommand { get; }

        public ReactiveCommand<Unit, Unit> MakeNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> LoadNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> SaveNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> MakeFilteredCommand { get; }

        public ReactiveCommand<Unit, Unit> SaveFilteredCommand { get; }

        private void LoadRaw()
        {
            try
            {
                this.ShowOpenFileDialog.Handle(Unit.Default).Subscribe(
                    x =>
                    {
                        try
                        {
                            if (x == null)
                            {
                                this.RawImage = null;
                            }
                            else
                            {
                                this.RawImage = new Bitmap(x);
                            }
                        }
                        catch
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch
            {
                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
            }
        }

        private async Task MakeNoisy()
        {
            try
            {
                this.IsProcessing = true;
                if (this.rawImage != null)
                {
                    var noisyBitmap = await Task.Run(() => this.noiseService.MakeNoisy(this.rawImage, this.sigma));
                    this.NoisyImage = noisyBitmap;
                    this.SelectedTab = 1;
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
            }
            finally
            {
                this.IsProcessing = false;
            }
        }

        private void LoadNoisy()
        {
            try
            {
                this.ShowOpenFileDialog.Handle(Unit.Default).Subscribe(
                    x =>
                    {
                        try
                        {
                            if (x == null)
                            {
                                this.NoisyImage = null;
                            }
                            else
                            {
                                this.NoisyImage = new Bitmap(x);
                            }
                        }
                        catch
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch
            {
                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
            }
        }

        private void SaveNoisy()
        {
            try
            {
                if (this.noisyImage != null)
                {
                    this.ShowSaveFileDialog.Handle(Unit.Default).Subscribe(
                        x =>
                        {
                            try
                            {
                                this.noisyImage.Save(x);
                            }
                            catch
                            {
                                Dispatcher.UIThread.InvokeAsync(() =>
                                {
                                    this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                                }).Wait();
                            }
                        },
                        x =>
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                            }).Wait();
                        },
                        () =>
                        {
                        });
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
            }
        }

        private async Task MakeFiltered()
        {
            try
            {
                this.IsProcessing = true;
                if (this.noisyImage != null)
                {
                    (var filteredBitmap, var filteringStats) = await Task.Run(() => this.filterService.MakeFiltered(this.noisyImage, this.sigma));
                    this.FilteredImage = filteredBitmap;
                    this.FilteringStats = filteringStats;
                    this.SelectedTab = 2;
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
            }
            finally
            {
                this.IsProcessing = false;
            }
        }

        private void SaveFiltered()
        {
            try
            {
                if (this.filteredImage != null)
                {
                    this.ShowSaveFileDialog.Handle(Unit.Default).Subscribe(
                        x =>
                        {
                            try
                            {
                                this.filteredImage.Save(x);
                            }
                            catch
                            {
                                Dispatcher.UIThread.InvokeAsync(() =>
                                {
                                    this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                                }).Wait();
                            }
                        },
                        x =>
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
                            }).Wait();
                        },
                        () =>
                        {
                        });
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B³¹d!").Subscribe();
            }
        }
    }
}
