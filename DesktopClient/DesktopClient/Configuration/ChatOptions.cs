namespace DesktopClient.Configuration;

/// <summary>
/// Chat validation limits. Values are loaded from appsettings.json "Chat" section.
/// </summary>
public sealed class ChatOptions
{
    public const string SectionName = "Chat";

    public int MinContentLength { get; set; }
    public int MaxContentLength { get; set; }
}
