using Data.Models;

namespace Service.Models;

/// <summary>
/// Business-layer model for a chat message, used by the service and web API.
/// Mirrors <see cref="DbChatMessage"/> with conversions to/from the DB and FastAPI.
/// </summary>
public class BusinessChatMessage
{
    public int Id { get; set; }
    public ChatRole Role { get; set; }
    public string Content { get; set; }
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Converts this instance to a <see cref="DbChatMessage"/> for persisting in the DB layer.
    /// If <see cref="CreatedAt"/> is not filled (default), sets it to <see cref="DateTime.Now"/> for correct sorting in the DB.
    /// </summary>
    public DbChatMessage ToDbChatMessage()
    {
        var createdAt = CreatedAt == default ? DateTime.Now : CreatedAt;
        return new DbChatMessage
        {
            Id = Id,
            Role = Role,
            Content = Content ?? string.Empty,
            CreatedAt = createdAt
        };
    }

    /// <summary>
    /// Creates a <see cref="BusinessChatMessage"/> from a <see cref="DbChatMessage"/> (e.g. from DB).
    /// </summary>
    public static BusinessChatMessage FromDbChatMessage(DbChatMessage message)
    {
        if (message == null)
            throw new ArgumentNullException(nameof(message));
        return new BusinessChatMessage
        {
            Id = message.Id,
            Role = message.Role,
            Content = message.Content ?? string.Empty,
            CreatedAt = message.CreatedAt
        };
    }

    /// <summary>
    /// Creates a <see cref="BusinessChatMessage"/> from the FastAPI answer string (assistant response).
    /// Used for saving the response to the DB and returning it to the web API layer.
    /// </summary>
    public static BusinessChatMessage FromAnswer(string answer)
    {
        return new BusinessChatMessage
        {
            Role = ChatRole.Assistant,
            Content = answer ?? string.Empty,
            CreatedAt = DateTime.UtcNow
        };
    }
}
