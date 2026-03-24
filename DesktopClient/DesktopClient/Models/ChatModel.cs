namespace DesktopClient.Models;

public sealed class ChatModel
{
    public ChatRole Role { get; init; }
    public string Content { get; init; } = string.Empty;
    public DateTime Timestamp { get; init; } = DateTime.Now;
}

public enum ChatRole
{
    Assistant,
    User
}