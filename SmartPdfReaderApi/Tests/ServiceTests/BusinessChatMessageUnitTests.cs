using Data.Models;
using Service.Models;
using Xunit;

namespace ServiceTests;

/// <summary>
/// Unit tests for <see cref="BusinessChatMessage"/> conversion methods.
/// </summary>
public class BusinessChatMessageUnitTests
{
    [Fact]
    public void FromDbChatMessage_Throws_When_Message_Is_Null()
    {
        Assert.Throws<ArgumentNullException>(() => BusinessChatMessage.FromDbChatMessage(null!));
    }

    [Fact]
    public void FromDbChatMessage_Maps_All_Properties()
    {
        var created = new DateTime(2025, 2, 19, 10, 0, 0, DateTimeKind.Utc);
        var db = new DbChatMessage
        {
            Id = 42,
            Role = ChatRole.Assistant,
            Content = "Hello back",
            CreatedAt = created
        };

        var business = BusinessChatMessage.FromDbChatMessage(db);

        Assert.Equal(42, business.Id);
        Assert.Equal(ChatRole.Assistant, business.Role);
        Assert.Equal("Hello back", business.Content);
        Assert.Equal(created, business.CreatedAt);
    }

    [Fact]
    public void FromAnswer_Returns_Assistant_With_Content_And_UtcNow()
    {
        var before = DateTime.UtcNow;
        var result = BusinessChatMessage.FromAnswer("The answer is 42.");
        var after = DateTime.UtcNow;

        Assert.Equal(ChatRole.Assistant, result.Role);
        Assert.Equal("The answer is 42.", result.Content);
        Assert.True(result.CreatedAt >= before && result.CreatedAt <= after.AddSeconds(1));
    }

    [Fact]
    public void FromAnswer_Handles_Null_Answer()
    {
        var result = BusinessChatMessage.FromAnswer(null!);
        Assert.Equal(string.Empty, result.Content);
    }

    [Fact]
    public void ToDbChatMessage_Maps_All_Properties_When_CreatedAt_Set()
    {
        var created = new DateTime(2025, 2, 19, 12, 0, 0);
        var business = new BusinessChatMessage
        {
            Id = 1,
            Role = ChatRole.User,
            Content = "Hi",
            CreatedAt = created
        };

        var db = business.ToDbChatMessage();

        Assert.Equal(1, db.Id);
        Assert.Equal(ChatRole.User, db.Role);
        Assert.Equal("Hi", db.Content);
        Assert.Equal(created, db.CreatedAt);
    }

    [Fact]
    public void ToDbChatMessage_Sets_CreatedAt_To_Now_When_Default()
    {
        var business = new BusinessChatMessage
        {
            Id = 0,
            Role = ChatRole.User,
            Content = "Question",
            CreatedAt = default
        };

        var before = DateTime.Now;
        var db = business.ToDbChatMessage();
        var after = DateTime.Now;

        Assert.True(db.CreatedAt >= before.AddSeconds(-1) && db.CreatedAt <= after.AddSeconds(1));
        Assert.Equal("Question", db.Content);
    }

    [Fact]
    public void ToDbChatMessage_Handles_Null_Content()
    {
        var business = new BusinessChatMessage { Content = null!, CreatedAt = DateTime.UtcNow };
        var db = business.ToDbChatMessage();
        Assert.Equal(string.Empty, db.Content);
    }
}
