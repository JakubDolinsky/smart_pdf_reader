namespace DesktopClient.Configuration;

public sealed class ApiOptions
{
    public const string SectionName = "Api";

    /// <summary>
    /// HTTP client timeout in seconds. Default 1800 (30 minutes) for long-running RAG/AskQuestion calls.
    /// Keep in sync manually with SmartPdfReaderApi/appsettings.json "DesktopClient": "TimeoutSeconds".
    /// </summary>
    public int TimeoutSeconds { get; set; } = 1800;

    public string BaseUrl { get; set; } = "http://localhost:5000";
}

