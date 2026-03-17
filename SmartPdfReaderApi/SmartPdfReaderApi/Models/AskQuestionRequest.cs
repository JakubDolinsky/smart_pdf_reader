using System.ComponentModel.DataAnnotations;
using Data.Models;

namespace SmartPdfReaderApi.Models;

/// <summary>
/// Request model for asking a question (user message).
/// </summary>
public class AskQuestionRequest
{
    /// <summary>Role of the speaker (e.g. User).</summary>
    [Required]
    public ChatRole Role { get; set; }

    /// <summary>Message content. Validated for min length by attribute; max length is validated in controller from config.</summary>
    [Required(ErrorMessage = "Content is required.")]
    [MinLength(1, ErrorMessage = "Content must be at least 1 character.")]
    public string Content { get; set; } = string.Empty;

    /// <summary>When the message was created (optional; defaults to server time if not set).</summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Converts this request to a <see cref="Service.Models.BusinessChatMessage"/> for the service layer.
    /// </summary>
    public Service.Models.BusinessChatMessage ToBusinessChatMessage()
    {
        return new Service.Models.BusinessChatMessage
        {
            Role = Role,
            Content = Content ?? string.Empty,
            CreatedAt = CreatedAt == default ? DateTime.UtcNow : CreatedAt
        };
    }
}
