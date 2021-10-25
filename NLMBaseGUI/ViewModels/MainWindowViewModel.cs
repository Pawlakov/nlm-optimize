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
    using Avalonia.Collections;
    using Avalonia.Threading;
    using NLMBaseGUI.Models;
    using NLMBaseGUI.NLM;
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
        private IImplementation implementation;
        private FilteringStatsModel? noisingStats;
        private FilteringStatsModel? filteringStats;

        public MainWindowViewModel()
        {
            this.noiseService = new NoiseService();
            this.filterService = new FilterService();

            var defaultImplementation = new DefaultImplementation();
            this.implementation = defaultImplementation;
            this.ImplementationOptions = new AvaloniaList<IImplementation> { defaultImplementation, };

            this.ShowOpenLibraryDialog = new Interaction<Unit, string[]?>();
            this.ShowOpenImageDialog = new Interaction<Unit, string?>();
            this.ShowSaveImageDialog = new Interaction<Unit, string?>();
            this.ShowMessageBox = new Interaction<string, Unit>();

            this.LoadRawCommand = ReactiveCommand.Create(this.LoadRaw);
            this.MakeNoisyCommand = ReactiveCommand.CreateFromTask(this.MakeNoisy, this.WhenAnyValue(x => x.RawImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.LoadNoisyCommand = ReactiveCommand.Create(this.LoadNoisy);
            this.SaveNoisyCommand = ReactiveCommand.Create(this.SaveNoisy, this.WhenAnyValue(x => x.NoisyImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.MakeFilteredCommand = ReactiveCommand.CreateFromTask(this.MakeFiltered, this.WhenAnyValue(x => x.NoisyImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.SaveFilteredCommand = ReactiveCommand.Create(this.SaveFiltered, this.WhenAnyValue(x => x.FilteredImage, (Bitmap? x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.LoadImplementationCommand = ReactiveCommand.Create(this.LoadImplementation);
            this.CancelTaskCommand = ReactiveCommand.Create(this.CancelTask);
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

                this.RaisePropertyChanged();
            }
        }

        public IImplementation Implementation
        {
            get => this.implementation;
            set => this.RaiseAndSetIfChanged(ref this.implementation, value);
        }

        public FilteringStatsModel? NoisingStats
        {
            get => this.noisingStats;
            set => this.RaiseAndSetIfChanged(ref this.noisingStats, value);
        }

        public FilteringStatsModel? FilteringStats 
        {
            get => this.filteringStats;
            set => this.RaiseAndSetIfChanged(ref this.filteringStats, value);
        }

        public AvaloniaList<IImplementation> ImplementationOptions { get; }

        public Interaction<Unit, string[]?> ShowOpenLibraryDialog { get; }

        public Interaction<Unit, string?> ShowOpenImageDialog { get; }

        public Interaction<Unit, string?> ShowSaveImageDialog { get; }

        public Interaction<string, Unit> ShowMessageBox { get; }

        public ReactiveCommand<Unit, Unit> LoadRawCommand { get; }

        public ReactiveCommand<Unit, Unit> MakeNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> LoadNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> SaveNoisyCommand { get; }

        public ReactiveCommand<Unit, Unit> MakeFilteredCommand { get; }

        public ReactiveCommand<Unit, Unit> SaveFilteredCommand { get; }

        public ReactiveCommand<Unit, Unit> LoadImplementationCommand { get; }

        public ReactiveCommand<Unit, Unit> CancelTaskCommand { get; }

        private void LoadRaw()
        {
            try
            {
                this.ShowOpenImageDialog.Handle(Unit.Default).Subscribe(
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
                                this.ShowMessageBox.Handle("B��d!").Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowMessageBox.Handle("B��d!").Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch
            {
                this.ShowMessageBox.Handle("B��d!").Subscribe();
            }
        }

        private async Task MakeNoisy()
        {
            try
            {
                this.IsProcessing = true;
                if (this.rawImage != null)
                {
                    (var noisyBitmap, var noisingStats) = await Task.Run(() => this.noiseService.MakeNoisy(this.rawImage, this.sigma));
                    this.NoisyImage = noisyBitmap;
                    this.NoisingStats = noisingStats;
                    this.SelectedTab = 1;
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B��d!").Subscribe();
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
                this.ShowOpenImageDialog.Handle(Unit.Default).Subscribe(
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
                                this.ShowMessageBox.Handle("B��d!").Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowMessageBox.Handle("B��d!").Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch
            {
                this.ShowMessageBox.Handle("B��d!").Subscribe();
            }
        }

        private void SaveNoisy()
        {
            try
            {
                if (this.noisyImage != null)
                {
                    this.ShowSaveImageDialog.Handle(Unit.Default).Subscribe(
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
                                    this.ShowMessageBox.Handle("B��d!").Subscribe();
                                }).Wait();
                            }
                        },
                        x =>
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowMessageBox.Handle("B��d!").Subscribe();
                            }).Wait();
                        },
                        () =>
                        {
                        });
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B��d!").Subscribe();
            }
        }

        private async Task MakeFiltered()
        {
            try
            {
                this.IsProcessing = true;
                if (this.noisyImage != null)
                {
                    (var filteredBitmap, var filteringStats) = await Task.Run(() => this.filterService.MakeFiltered(this.implementation, this.rawImage, this.noisyImage, this.sigma));
                    this.FilteredImage = filteredBitmap;
                    this.FilteringStats = filteringStats;
                    this.SelectedTab = 2;
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B��d!").Subscribe();
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
                    this.ShowSaveImageDialog.Handle(Unit.Default).Subscribe(
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
                                    this.ShowMessageBox.Handle("B��d!").Subscribe();
                                }).Wait();
                            }
                        },
                        x =>
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowMessageBox.Handle("B��d!").Subscribe();
                            }).Wait();
                        },
                        () =>
                        {
                        });
                }
            }
            catch
            {
                this.ShowMessageBox.Handle("B��d!").Subscribe();
            }
        }

        private void LoadImplementation()
        {
            try
            {
                this.ShowOpenLibraryDialog.Handle(Unit.Default).Subscribe(
                    x => 
                    {
                        try
                        {
                            if (x != null)
                            {
                                foreach (var libraryName in x)
                                {
                                    var file = new FileInfo(libraryName);
                                    if (file.Exists)
                                    {
                                        var implementation = new ExternalImplementation(file);
                                        this.ImplementationOptions.Add(implementation);
                                        this.Implementation = implementation;
                                    }
                                }
                            }
                        }
                        catch
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowMessageBox.Handle("B��d!").Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowMessageBox.Handle("B��d!").Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch
            {
                this.ShowMessageBox.Handle("B��d!").Subscribe();
            }
        }

        private void CancelTask()
        {

        }
    }
}