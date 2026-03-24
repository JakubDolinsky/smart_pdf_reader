using Data.Models;
using Data.Repository;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Service.Configuration;
using Service.Models;

namespace Service.Services;

/// <summary>
/// Orchestrates chat: loads conversation, asks the RAG API (with question + last 4 messages), persists questions and answers.
/// Validates question length and supports clearing the conversation.
/// </summary>
public class ChatMessageService
{
    private readonly IRepository _repository;
    private readonly Clients.IFastApiClient _fastApiClient;
    private readonly int _maxQuestionLength;
    private readonly ILogger<ChatMessageService> _logger;
    private readonly int _countOfSelectedMessagesForRequest;

    public ChatMessageService(
        IRepository repository,
        Clients.IFastApiClient fastApiClient,
        IOptions<ChatServiceOptions> options,
        ILogger<ChatMessageService> logger)
    {
        _repository = repository ?? throw new ArgumentNullException(nameof(repository));
        _fastApiClient = fastApiClient ?? throw new ArgumentNullException(nameof(fastApiClient));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _maxQuestionLength = options?.Value?.MaxQuestionLength ?? 2000;
        if (_maxQuestionLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxQuestionLength must be positive.");
        _countOfSelectedMessagesForRequest = options?.Value?.CountOfLastMessagesForRequest ?? 2000;
        if (_countOfSelectedMessagesForRequest <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "CountOfSelectedMessagesForRequest must be positive.");
    }

    /// <summary>
    /// Loads the whole conversation (all messages) in chronological order.
    /// </summary>
    public async Task<IReadOnlyList<BusinessChatMessage>> LoadConversationAsync(CancellationToken cancellationToken = default)
    {
        var messages = await _repository.GetAllMessagesAsync(cancellationToken).ConfigureAwait(false);
        var result = messages.Select(BusinessChatMessage.FromDbChatMessage).ToList();
        _logger.LogDebug("LoadConversationAsync returning {Count} messages", result.Count);
        return result;
    }

    /// <summary>
    /// Validates question content (must be non-empty) and length (content length &lt;= MaxQuestionLength).
    /// </summary>
    /// <exception cref="ArgumentNullException">When question is null.</exception>
    /// <exception cref="ArgumentException">When content is null/empty or exceeds max length.</exception>
    public void ValidateQuestionLength(BusinessChatMessage question)
    {
        if (question == null)
            throw new ArgumentNullException(nameof(question));

        var content = question.Content ?? string.Empty;
        if (string.IsNullOrEmpty(content))
        {
            _logger.LogWarning("Question content is null or empty.");
            throw new ArgumentException("Question content must not be null or empty.", nameof(question));
        }

        if (content.Length > _maxQuestionLength)
        {
            _logger.LogWarning("Question length {Length} exceeds maximum {Max}", content.Length, _maxQuestionLength);
            throw new ArgumentException(
                $"Question length ({content.Length}) exceeds maximum allowed ({_maxQuestionLength} characters).",
                nameof(question));
        }
    }

    /// <summary>
    /// Inserts the question into the DB, loads the last 3 messages, sends the newest question + the other 2 to the RAG FastAPI,
    /// converts the answer to <see cref="BusinessChatMessage"/>, saves it to the DB, and returns it.
    /// Validates question length before processing.
    /// </summary>
    public async Task<BusinessChatMessage> AskAsync(BusinessChatMessage question, CancellationToken cancellationToken = default)
    {
        if (question == null)
            throw new ArgumentNullException(nameof(question));

        ValidateQuestionLength(question);
        _logger.LogInformation("AskAsync: processing question (length={Length})", (question.Content ?? string.Empty).Length);

        var chatMessage = question.ToDbChatMessage();
        chatMessage.Role = ChatRole.User;
        await _repository.InsertAsync(chatMessage, cancellationToken).ConfigureAwait(false);

        var lastMessages = await _repository.GetMessagesAsync(_countOfSelectedMessagesForRequest, cancellationToken).ConfigureAwait(false);
        var businessMessages = lastMessages.Select(BusinessChatMessage.FromDbChatMessage).ToList();

        string currentQuestion;
        IReadOnlyList<BusinessChatMessage> lastMessagesForReq;

        if (businessMessages.Count >= 1)
        {
            var newest = businessMessages[businessMessages.Count - 1];
            currentQuestion = newest.Content ?? string.Empty;
            lastMessagesForReq = businessMessages.Take(businessMessages.Count - 1).ToList();
        }
        else
        {
            currentQuestion = question.Content ?? string.Empty;
            lastMessagesForReq = Array.Empty<BusinessChatMessage>();
        }

        _logger.LogDebug("AskAsync: calling RAG with history count={Count}", lastMessagesForReq.Count);
        var answerText = await _fastApiClient
            .GetAnswerAsync(currentQuestion, lastMessagesForReq, cancellationToken)
            .ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(answerText))
        {
            _logger.LogWarning("AskAsync: RAG returned null or empty answer.");
            throw new InvalidOperationException("RAG FastAPI returned null or empty answer.");
        }

        _logger.LogInformation("AskAsync: RAG returned answer (length={Length})", answerText.Length);

        var answerModel = BusinessChatMessage.FromAnswer(answerText);
        var answerToSave = answerModel.ToDbChatMessage();
        await _repository.InsertAsync(answerToSave, cancellationToken).ConfigureAwait(false);

        return new BusinessChatMessage
        {
            Id = answerToSave.Id,
            Role = ChatRole.Assistant,
            Content = answerToSave.Content,
            CreatedAt = answerToSave.CreatedAt
        };
    }

    /// <summary>
    /// Deletes the whole message collection from the DB.
    /// </summary>
    public async Task DeleteAllMessagesAsync(CancellationToken cancellationToken = default)
    {
        await _repository.DeleteAllAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("DeleteAllMessagesAsync completed");
    }
}
