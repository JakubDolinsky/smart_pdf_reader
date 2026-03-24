using DesktopClient.Models;
using ApiChatRole = DesktopClient.ModelsDTO.ChatRole;
using ClientChatRole = DesktopClient.Models.ChatRole;

namespace DesktopClient.ModelsDTO;

/// <summary>
/// Maps generated API DTOs into the existing DesktopClient domain models.
/// </summary>
public static class ChatDtoMapper
{
    public static ChatModel ToChatModel(ChatMessageResponse dto)
    {
        if (dto is null)
            throw new ArgumentNullException(nameof(dto));

        return new ChatModel
        {
            Role = ToClientRole(dto.Role),
            Content = dto.Content ?? string.Empty,
            // API returns UTC; convert to local time so response time appears after question time in the UI.
            Timestamp = dto.CreatedAt.LocalDateTime
        };
    }

    public static ClientChatRole ToClientRole(ApiChatRole apiRole) =>
        apiRole switch
        {
            // In API: Data.Models.ChatRole enum order is User (0), Assistant (1).
            ApiChatRole._0 => ClientChatRole.User,
            ApiChatRole._1 => ClientChatRole.Assistant,
            _ => ClientChatRole.User
        };

    public static ApiChatRole ToApiRole(ClientChatRole clientRole) =>
        clientRole switch
        {
            ClientChatRole.User => ApiChatRole._0,
            ClientChatRole.Assistant => ApiChatRole._1,
            _ => ApiChatRole._0
        };

    public static AskQuestionRequest ToAskQuestionRequest(
        string content,
        ClientChatRole clientRole,
        DateTime? createdAtUtc = null)
    {
        var createdAt = createdAtUtc ?? DateTime.UtcNow;

        return new AskQuestionRequest
        {
            Content = content ?? string.Empty,
            Role = ToApiRole(clientRole),
            CreatedAt = new DateTimeOffset(DateTime.SpecifyKind(createdAt, DateTimeKind.Utc))
        };
    }
}

