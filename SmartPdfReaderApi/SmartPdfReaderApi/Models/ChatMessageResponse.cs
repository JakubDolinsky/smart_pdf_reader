using Data.Models;
using Service.Models;

namespace SmartPdfReaderApi.Models;

/// <summary>
/// Response model for a single chat message (question or answer).
/// </summary>
public class ChatMessageResponse
{
    /// <summary>Message id from the database.</summary>
    public int Id { get; set; }

    /// <summary>Role of the speaker (User or Assistant).</summary>
    public ChatRole Role { get; set; }

    /// <summary>Message content.</summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>When the message was created.</summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Creates a response from a <see cref="BusinessChatMessage"/>.
    /// </summary>
    public static ChatMessageResponse FromBusinessChatMessage(BusinessChatMessage message)
    {
        if (message == null)
            throw new ArgumentNullException(nameof(message));
        return new ChatMessageResponse
        {
            Id = message.Id,
            Role = message.Role,
            Content = message.Content ?? string.Empty,
            CreatedAt = message.CreatedAt
        };
    }
}
