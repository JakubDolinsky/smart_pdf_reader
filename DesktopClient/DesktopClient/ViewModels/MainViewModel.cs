using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Windows.Threading;
using DesktopClient.Client;
using DesktopClient.Configuration;
using DesktopClient.Models;
using ClientChatRole = DesktopClient.Models.ChatRole;
using DesktopClient.ModelsDTO;
using Serilog;

namespace DesktopClient.ViewModels;

public sealed class MainViewModel : INotifyPropertyChanged
{
    private readonly ChatOptions _chatOptions;
    private readonly ISmartPdfReaderApiClient _apiClient;
    private bool _isBusy;
    private string _statusText = string.Empty;
    private DateTime? _busyStart;
    private string _busyPrefix = string.Empty;
    private bool _preserveStatusText;
    private readonly DispatcherTimer _busyTimer;

    public int MinContentLength => _chatOptions.MinContentLength;
    public int MaxContentLength => _chatOptions.MaxContentLength;

    public ObservableCollection<ChatModel> Messages { get; } = new();

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (_isBusy == value) return;
            _isBusy = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(IsSubmitButtonEnabled));
            OnPropertyChanged(nameof(IsDeleteButtonEnabled));
            SubmitCommand.RaiseCanExecuteChanged();
            DeleteConversationCommand.RaiseCanExecuteChanged();
        }
    }

    public string StatusText
    {
        get => _statusText;
        private set
        {
            if (_statusText == value) return;
            _statusText = value;
            OnPropertyChanged();
        }
    }

    public bool IsSubmitButtonEnabled => IsSubmitEnabled && !IsBusy;
    public bool IsDeleteButtonEnabled => IsDeleteEnabled && !IsBusy;

    private string _inputText = string.Empty;
    public string InputText
    {
        get => _inputText;
        set
        {
            if (_inputText == value) return;
            _inputText = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(IsSubmitEnabled));
            OnPropertyChanged(nameof(IsSubmitButtonEnabled));
            SubmitCommand.RaiseCanExecuteChanged();
        }
    }

    public bool IsSubmitEnabled => InputText.Length >= MinContentLength && InputText.Length <= MaxContentLength;
    public bool IsDeleteEnabled => Messages.Count > 0;

    public AsyncRelayCommand SubmitCommand { get; }
    public AsyncRelayCommand DeleteConversationCommand { get; }

    public MainViewModel(ChatOptions chatOptions, ApiOptions apiOptions)
    {
        _chatOptions = chatOptions ?? throw new ArgumentNullException(nameof(chatOptions));
        if (apiOptions is null)
            throw new ArgumentNullException(nameof(apiOptions));

        if (string.IsNullOrWhiteSpace(apiOptions.BaseUrl))
            throw new ArgumentException("API base URL must be provided in ApiOptions.", nameof(apiOptions));

        var httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(apiOptions.TimeoutSeconds)
        };
        _apiClient = new SmartPdfReaderApiClient(apiOptions.BaseUrl, httpClient);

        SubmitCommand = new AsyncRelayCommand(SubmitAsync, () => IsSubmitButtonEnabled);
        DeleteConversationCommand = new AsyncRelayCommand(DeleteConversationAsync, () => IsDeleteButtonEnabled);

        _busyTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _busyTimer.Tick += (_, _) =>
        {
            if (_busyStart is null)
                return;

            // If we preserved a manual message (error), don't overwrite it.
            if (_preserveStatusText)
                return;

            var elapsed = DateTime.Now - _busyStart.Value;
            var formatted = elapsed.ToString(@"hh\:mm\:ss");

            StatusText = string.IsNullOrWhiteSpace(_busyPrefix)
                ? StatusText
                : $"{_busyPrefix} ({formatted})";
        };

        Messages.CollectionChanged += (_, __) =>
        {
            OnPropertyChanged(nameof(IsDeleteEnabled));
            OnPropertyChanged(nameof(IsDeleteButtonEnabled));
            DeleteConversationCommand.RaiseCanExecuteChanged();
        };
    }

    public async Task LoadConversationAsync()
    {
        Log.Information("Loading conversation (min={Min}, max={Max})", MinContentLength, MaxContentLength);
        StartBusy("Loading conversation");
        try
        {
            Messages.Clear();

            var dtos = await _apiClient.LoadAllMessagesAsync();
            foreach (var dto in dtos)
                Messages.Add(ChatDtoMapper.ToChatModel(dto));

            Log.Information("Loaded conversation messages={Count}", Messages.Count);
        }
        catch (Exception ex)
        {
            _preserveStatusText = true;
            // Keep client usable even if the API is not reachable.
            StatusText = $"Failed to load conversation: {ex.Message}";
            Log.Error(ex, "Failed to load conversation.");
        }
        finally
        {
            StopBusy();
        }
    }

    private async Task SubmitAsync()
    {
        if (!IsSubmitEnabled) return;

        Log.Information("Submitting question. Length={Length}", InputText.Length);
        StartBusy("Waiting for response");

        var userText = InputText;

        Messages.Add(new ChatModel
        {
            Role = ClientChatRole.User,
            Content = userText,
            Timestamp = DateTime.Now
        });

        InputText = string.Empty;

        try
        {
            var request = ChatDtoMapper.ToAskQuestionRequest(userText, ClientChatRole.User);
            var responseDto = await _apiClient.AskQuestionAsync(request);

            Messages.Add(ChatDtoMapper.ToChatModel(responseDto));
            Log.Information("Received assistant response. ResponseId={Id}, Length={Length}",
                responseDto.Id, responseDto.Content?.Length ?? 0);
        }
        catch (Exception ex)
        {
            _preserveStatusText = true;
            StatusText = $"Submit failed: {ex.Message}";
            Messages.Add(new ChatModel
            {
                Role = ClientChatRole.Assistant,
                Content = $"(error) {ex.Message}",
                Timestamp = DateTime.Now
            });
            Log.Error(ex, "Submit failed.");
        }
        finally
        {
            StopBusy();
        }
    }

    private async Task DeleteConversationAsync()
    {
        if (!IsDeleteEnabled)
            return;

        Log.Information("Deleting conversation. CurrentCount={Count}", Messages.Count);
        StartBusy("Deleting conversation");

        try
        {
            await _apiClient.DeleteAllAsync();
        }
        catch (Exception ex)
        {
            _preserveStatusText = true;
            StatusText = $"Failed to delete conversation: {ex.Message}";
            Log.Error(ex, "Failed to delete conversation.");
        }
        finally
        {
            Messages.Clear();
            StopBusy();
        }
    }

    private void StartBusy(string prefix)
    {
        _preserveStatusText = false;
        _busyPrefix = prefix ?? string.Empty;
        _busyStart = DateTime.Now;

        StatusText = _busyPrefix;
        IsBusy = true;
        _busyTimer.Start();
    }

    private void StopBusy()
    {
        _busyTimer.Stop();
        _busyStart = null;
        _busyPrefix = string.Empty;

        IsBusy = false;

        if (!_preserveStatusText)
            StatusText = string.Empty;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}

