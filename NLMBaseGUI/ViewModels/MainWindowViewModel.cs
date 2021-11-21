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

    public class MainWindowViewModel
        : ViewModelBase
    {
        private readonly NoiseService noiseService;
        private readonly FilterService filterService;

        private bool isProcessing;
        private int selectedTab;
        private Bitmap rawImage;
        private Bitmap noisyImage;
        private Bitmap filteredImage;
        private int sigma;
        private ImplementationModel implementation;
        private FilteringStatsModel noisingStats;
        private FilteringStatsModel filteringStats;
        private ISession currentSesstion;

        public MainWindowViewModel()
        {
            this.noiseService = new NoiseService();
            this.filterService = new FilterService();

            this.sigma = 1;
            var defaultImplementation = new ImplementationModel(null);
            this.implementation = defaultImplementation;
            this.ImplementationOptions = new AvaloniaList<ImplementationModel> { defaultImplementation, };

            this.ShowOpenLibraryDialog = new Interaction<Unit, string[]>();
            this.ShowOpenImageDialog = new Interaction<Unit, string>();
            this.ShowSaveImageDialog = new Interaction<Unit, string>();
            this.ShowExceptionMessageBox = new Interaction<Exception, Unit>();

            this.LoadRawCommand = ReactiveCommand.Create(this.LoadRaw);
            this.MakeNoisyCommand = ReactiveCommand.CreateFromTask(this.MakeNoisy, this.WhenAnyValue(x => x.RawImage, (Bitmap x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.LoadNoisyCommand = ReactiveCommand.Create(this.LoadNoisy);
            this.SaveNoisyCommand = ReactiveCommand.Create(this.SaveNoisy, this.WhenAnyValue(x => x.NoisyImage, (Bitmap x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.MakeFilteredCommand = ReactiveCommand.CreateFromTask(this.MakeFiltered, this.WhenAnyValue(x => x.NoisyImage, (Bitmap x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.SaveFilteredCommand = ReactiveCommand.Create(this.SaveFiltered, this.WhenAnyValue(x => x.FilteredImage, (Bitmap x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
            this.LoadImplementationCommand = ReactiveCommand.Create(this.LoadImplementation);
            this.CancelTaskCommand = ReactiveCommand.Create(this.CancelTask, this.WhenAnyValue(x => x.CurrentSesstion, (ISession x) => x != null).ObserveOn(RxApp.MainThreadScheduler));
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
                    else if (parsed < 1)
                    {
                        parsed = 1;
                    }

                    this.sigma = parsed;
                }

                this.RaisePropertyChanged();
            }
        }

        public ImplementationModel Implementation
        {
            get => this.implementation;
            set => this.RaiseAndSetIfChanged(ref this.implementation, value);
        }

        public FilteringStatsModel NoisingStats
        {
            get => this.noisingStats;
            set => this.RaiseAndSetIfChanged(ref this.noisingStats, value);
        }

        public FilteringStatsModel FilteringStats
        {
            get => this.filteringStats;
            set => this.RaiseAndSetIfChanged(ref this.filteringStats, value);
        }

        public ISession CurrentSesstion
        {
            get => this.currentSesstion;
            set => this.RaiseAndSetIfChanged(ref this.currentSesstion, value);
        }

        public AvaloniaList<ImplementationModel> ImplementationOptions { get; }

        public Interaction<Unit, string[]> ShowOpenLibraryDialog { get; }

        public Interaction<Unit, string> ShowOpenImageDialog { get; }

        public Interaction<Unit, string> ShowSaveImageDialog { get; }

        public Interaction<Exception, Unit> ShowExceptionMessageBox { get; }

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
                            if (x != null)
                            {
                                this.RawImage = new Bitmap(x);
                            }
                        }
                        catch (Exception exception)
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowExceptionMessageBox.Handle(x).Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch (Exception exception)
            {
                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
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
            catch (Exception exception)
            {
                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
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
                            if (x != null)
                            {
                                this.NoisyImage = new Bitmap(x);
                            }
                        }
                        catch (Exception exception)
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowExceptionMessageBox.Handle(x).Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch (Exception exception)
            {
                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
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
                                if (x != null)
                                {
                                    this.noisyImage.Save(x);
                                }
                            }
                            catch (Exception exception)
                            {
                                Dispatcher.UIThread.InvokeAsync(() =>
                                {
                                    this.ShowExceptionMessageBox.Handle(exception).Subscribe();
                                }).Wait();
                            }
                        },
                        x =>
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowExceptionMessageBox.Handle(x).Subscribe();
                            }).Wait();
                        },
                        () =>
                        {
                        });
                }
            }
            catch (Exception exception)
            {
                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
            }
        }

        private async Task MakeFiltered()
        {
            try
            {
                this.IsProcessing = true;
                if (this.noisyImage != null)
                {
                    this.CurrentSesstion = this.filterService.SetUp(this.implementation, this.noisyImage, this.sigma);
                    (var filteredBitmap, var filteringStats) = await this.CurrentSesstion.Run(this.rawImage);
                    this.FilteredImage = filteredBitmap;
                    this.FilteringStats = filteringStats;
                    this.SelectedTab = 2;
                }
            }
            catch (Exception exception)
            {
                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
            }
            finally
            {
                this.CurrentSesstion = null;
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
                                if (x != null)
                                {
                                    this.filteredImage.Save(x);
                                }
                            }
                            catch (Exception exception)
                            {
                                Dispatcher.UIThread.InvokeAsync(() =>
                                {
                                    this.ShowExceptionMessageBox.Handle(exception).Subscribe();
                                }).Wait();
                            }
                        },
                        x =>
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowExceptionMessageBox.Handle(x).Subscribe();
                            }).Wait();
                        },
                        () =>
                        {
                        });
                }
            }
            catch (Exception exception)
            {
                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
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
                                        var implementation = new ImplementationModel(file);
                                        this.ImplementationOptions.Add(implementation);
                                        this.Implementation = implementation;
                                    }
                                }
                            }
                        }
                        catch (Exception exception)
                        {
                            Dispatcher.UIThread.InvokeAsync(() =>
                            {
                                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
                            }).Wait();
                        }
                    },
                    x =>
                    {
                        Dispatcher.UIThread.InvokeAsync(() =>
                        {
                            this.ShowExceptionMessageBox.Handle(x).Subscribe();
                        }).Wait();
                    },
                    () =>
                    {
                    });
            }
            catch (Exception exception)
            {
                this.ShowExceptionMessageBox.Handle(exception).Subscribe();
            }
        }

        private void CancelTask()
        {
            if (this.CurrentSesstion != null)
            {
                this.CurrentSesstion.Cancel();
            }
        }
    }
}