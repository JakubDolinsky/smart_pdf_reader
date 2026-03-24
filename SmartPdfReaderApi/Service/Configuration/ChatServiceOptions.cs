namespace Service.Configuration;

/// <summary>
/// Configuration for the chat/RAG service. Values are loaded from appsettings.json "ChatService" section.
/// </summary>
public class ChatServiceOptions
{
    public const string SectionName = "ChatService";

    public int MinQuestionLength { get; set; } = 1;
    public int MaxQuestionLength { get; set; } = 2000;
    public string FastApiBaseUrl { get; set; } = string.Empty;
    public int FastApiTimeoutSeconds { get; set; }

    public int CountOfLastMessagesForRequest { get; set; } = 3;
}
