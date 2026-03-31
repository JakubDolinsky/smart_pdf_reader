namespace Data.Models
{
    /// <summary>
    /// Role of the speaker in a chat message.
    /// </summary>
    public enum ChatRole
    {
        User,
        Assistant
    }

    /// <summary>
    /// Data model for a single chat message (user or assistant) in a conversation.
    /// </summary>
    public class DbChatMessage
    {
        public int Id { get; set; }
        public ChatRole Role { get; set; }
        public string Content { get; set; }
        public DateTime CreatedAt { get; set; }
    }
}
