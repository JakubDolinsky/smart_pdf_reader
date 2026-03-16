using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Service.Models;

namespace Service.Clients;

/// <summary>
/// HTTP client for the RAG FastAPI: sends question + last 4 messages and returns the answer string.
/// </summary>
public class FastApiClient : IFastApiClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<FastApiClient> _logger;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true
    };

    public FastApiClient(HttpClient httpClient, ILogger<FastApiClient> logger)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Sends the question and last 4 previous messages to the RAG FastAPI and returns the answer string.
    /// </summary>
    /// <param name="question">Current user question.</param>
    /// <param name="lastFourMessages">Up to 4 previous messages (oldest first).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The answer string from the FastAPI.</returns>
    public async Task<string> GetAnswerAsync(
        string question,
        IReadOnlyList<BusinessChatMessage> lastFourMessages,
        CancellationToken cancellationToken = default)
    {
        var history = lastFourMessages
            .Select(m => new HistoryMessageDto
            {
                Role = m.Role == Data.Models.ChatRole.User ? "user" : "assistant",
                Content = m.Content ?? string.Empty
            })
            .ToList();

        var request = new AskRequestDto
        {
            Question = question?.Trim() ?? string.Empty,
            History = history
        };

        _logger.LogDebug("GetAnswerAsync: POST /ask question length={Length}, history count={Count}", request.Question.Length, history.Count);

        using var response = await _httpClient
            .PostAsJsonAsync("ask", request, JsonOptions, cancellationToken)
            .ConfigureAwait(false);

        response.EnsureSuccessStatusCode();

        var result = await response.Content
            .ReadFromJsonAsync<AskResponseDto>(JsonOptions, cancellationToken)
            .ConfigureAwait(false);

        var answer = result?.Answer ?? string.Empty;
        _logger.LogDebug("GetAnswerAsync: received answer length={Length}", answer.Length);
        return answer;
    }

    private sealed class HistoryMessageDto
    {
        public string Role { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
    }

    private sealed class AskRequestDto
    {
        public string Question { get; set; } = string.Empty;
        public List<HistoryMessageDto>? History { get; set; }
    }

    private sealed class AskResponseDto
    {
        public string Answer { get; set; } = string.Empty;
    }
}
