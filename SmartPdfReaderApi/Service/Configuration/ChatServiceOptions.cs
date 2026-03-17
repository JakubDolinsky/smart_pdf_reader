namespace Service.Configuration;

/// <summary>
/// Configuration for the chat/RAG service (max question length and FastAPI base URL).
/// </summary>
public class ChatServiceOptions
{
    public const string SectionName = "ChatService";

    /// <summary>
    /// Minimum allowed length (in characters) for a user question. Default 1.
    /// </summary>
    public int MinQuestionLength { get; set; } = 1;

    /// <summary>
    /// Maximum allowed length (in characters) for a user question. Default 2000.
    /// </summary>
    public int MaxQuestionLength { get; set; } = 2000;

    /// <summary>
    /// Base URL of the RAG FastAPI (e.g. http://localhost:8000).
    /// </summary>
    public string FastApiBaseUrl { get; set; } = "http://localhost:8000";

    /// <summary>
    /// Timeout in seconds for the HTTP request to the RAG FastAPI. RAG can take 10+ minutes; set accordingly (e.g. 900 = 15 minutes). Default 900.
    /// </summary>
    public int FastApiTimeoutSeconds { get; set; } = 1800;
}
