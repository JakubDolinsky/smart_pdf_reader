using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using Service.Configuration;
using Service.Services;
using SmartPdfReaderApi.Models;

namespace SmartPdfReaderApi.Controllers;

/// <summary>
/// API for chat: ask questions (RAG), load all messages, and clear conversation.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class ChatController : ControllerBase
{
    private readonly ChatMessageService _chatMessageService;
    private readonly ChatServiceOptions _options;
    private readonly ILogger<ChatController> _logger;

    public ChatController(
        ChatMessageService chatMessageService,
        IOptions<ChatServiceOptions> options,
        ILogger<ChatController> logger)
    {
        _chatMessageService = chatMessageService ?? throw new ArgumentNullException(nameof(chatMessageService));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Ask a question: persists the question, calls RAG, persists the answer and returns it.
    /// </summary>
    /// <param name="request">Question content, role and optional created-at.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Assistant response and 200 OK, or 400 if validation fails, 503 if RAG fails.</returns>
    [HttpPost("AskQuestion")]
    [ProducesResponseType(typeof(ChatMessageResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status503ServiceUnavailable)]
    public async Task<ActionResult<ChatMessageResponse>> AskQuestion(
        [FromBody] AskQuestionRequest request,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("AskQuestion called: Content length={Length}", request?.Content?.Length ?? 0);

        if (request == null)
        {
            _logger.LogWarning("AskQuestion: request body is null.");
            return BadRequest("Request body is required.");
        }

        var content = request.Content ?? string.Empty;
        if (content.Length < _options.MinQuestionLength)
        {
            _logger.LogWarning("AskQuestion: content length {Length} below minimum {Min}", content.Length, _options.MinQuestionLength);
            return BadRequest($"Content must be at least {_options.MinQuestionLength} character(s).");
        }

        if (content.Length > _options.MaxQuestionLength)
        {
            _logger.LogWarning("AskQuestion: content length {Length} exceeds maximum {Max}", content.Length, _options.MaxQuestionLength);
            return BadRequest($"Content must not exceed {_options.MaxQuestionLength} characters.");
        }

        try
        {
            var businessMessage = request.ToBusinessChatMessage();
            var answer = await _chatMessageService.AskAsync(businessMessage, cancellationToken).ConfigureAwait(false);
            var response = ChatMessageResponse.FromBusinessChatMessage(answer);
            _logger.LogInformation("AskQuestion completed: answer Id={Id}, Content length={Length}", response.Id, response.Content.Length);
            return Ok(response);
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "AskQuestion validation failed.");
            return BadRequest(ex.Message);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "AskQuestion failed: RAG returned invalid response.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable, ex.Message);
        }
    }

    /// <summary>
    /// Load all messages from the database (e.g. when the client starts).
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of all messages in chronological order.</returns>
    [HttpGet("LoadAllMessages")]
    [ProducesResponseType(typeof(IReadOnlyList<ChatMessageResponse>), StatusCodes.Status200OK)]
    public async Task<ActionResult<IReadOnlyList<ChatMessageResponse>>> LoadAllMessages(CancellationToken cancellationToken)
    {
        _logger.LogInformation("LoadAllMessages called.");

        var messages = await _chatMessageService.LoadConversationAsync(cancellationToken).ConfigureAwait(false);
        var response = messages.Select(ChatMessageResponse.FromBusinessChatMessage).ToList();
        _logger.LogInformation("LoadAllMessages completed: returned {Count} messages.", response.Count);
        return Ok(response);
    }

    /// <summary>
    /// Delete all messages in the conversation (cleanup whole collection).
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>204 No Content on success.</returns>
    [HttpDelete("DeleteAll")]
    [ProducesResponseType(StatusCodes.Status204NoContent)]
    public async Task<IActionResult> DeleteAll(CancellationToken cancellationToken)
    {
        _logger.LogInformation("DeleteAll called.");

        await _chatMessageService.DeleteAllMessagesAsync(cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("DeleteAll completed.");
        return NoContent();
    }
}
