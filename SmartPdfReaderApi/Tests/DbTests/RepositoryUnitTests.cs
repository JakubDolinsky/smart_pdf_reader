using Data.DataContext;
using Data.Models;
using Data.Repository;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace DbTests;

/// <summary>
/// Unit tests for <see cref="Repository"/> using an in-memory database.
/// </summary>
public class RepositoryUnitTests : IAsyncLifetime
{
    private ChatHistoryDbContext _context = null!;
    private Repository _repository = null!;
    private const int MaxMessageCount = 5;

    public async Task InitializeAsync()
    {
        var options = new DbContextOptionsBuilder<ChatHistoryDbContext>()
            .UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString())
            .Options;
        _context = new ChatHistoryDbContext(options);
        await _context.Database.EnsureCreatedAsync();
        _repository = new Repository(_context, MaxMessageCount, NullLogger<Repository>.Instance);
    }

    public Task DisposeAsync() => Task.CompletedTask;

    [Fact]
    public async Task InsertAsync_Throws_When_Message_Is_Null()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() => _repository.InsertAsync(null!));
    }

    [Fact]
    public async Task InsertAsync_Adds_Message_When_Under_Capacity()
    {
        var msg = NewMessage(ChatRole.User, "Hello");
        await _repository.InsertAsync(msg);
        var list = await _repository.GetMessagesAsync(10);
        Assert.Single(list);
        Assert.Equal("Hello", list[0].Content);
        Assert.Equal(ChatRole.User, list[0].Role);
    }

    [Fact]
    public async Task GetMessagesAsync_Returns_Empty_When_Count_Zero()
    {
        await _repository.InsertAsync(NewMessage(ChatRole.User, "Hi"));
        var list = await _repository.GetMessagesAsync(0);
        Assert.Empty(list);
    }

    [Fact]
    public async Task GetMessagesAsync_Returns_Empty_When_Count_Negative()
    {
        var list = await _repository.GetMessagesAsync(-1);
        Assert.Empty(list);
    }

    [Fact]
    public async Task GetMessagesAsync_Returns_Chronological_Order_Oldest_First()
    {
        await _repository.InsertAsync(NewMessage(ChatRole.User, "First", DateTime.UtcNow.AddMinutes(-2)));
        await _repository.InsertAsync(NewMessage(ChatRole.Assistant, "Second", DateTime.UtcNow.AddMinutes(-1)));
        await _repository.InsertAsync(NewMessage(ChatRole.User, "Third", DateTime.UtcNow));
        var list = await _repository.GetMessagesAsync(10);
        Assert.Equal(3, list.Count);
        Assert.Equal("First", list[0].Content);
        Assert.Equal("Second", list[1].Content);
        Assert.Equal("Third", list[2].Content);
    }

    [Fact]
    public async Task GetMessagesAsync_Returns_Only_Requested_Count_Last_Messages()
    {
        await _repository.InsertAsync(NewMessage(ChatRole.User, "1", DateTime.UtcNow.AddMinutes(-3)));
        await _repository.InsertAsync(NewMessage(ChatRole.User, "2", DateTime.UtcNow.AddMinutes(-2)));
        await _repository.InsertAsync(NewMessage(ChatRole.User, "3", DateTime.UtcNow.AddMinutes(-1)));
        await _repository.InsertAsync(NewMessage(ChatRole.User, "4", DateTime.UtcNow));
        var list = await _repository.GetMessagesAsync(2);
        Assert.Equal(2, list.Count);
        Assert.Equal("3", list[0].Content);
        Assert.Equal("4", list[1].Content);
    }

    [Fact]
    public async Task InsertAsync_Removes_Oldest_When_At_Capacity()
    {
        for (int i = 0; i < MaxMessageCount; i++)
            await _repository.InsertAsync(NewMessage(ChatRole.User, $"Msg{i}", DateTime.UtcNow.AddMinutes(-MaxMessageCount + i)));
        await _repository.InsertAsync(NewMessage(ChatRole.Assistant, "NewOne", DateTime.UtcNow.AddMinutes(1)));
        var list = await _repository.GetMessagesAsync(10);
        Assert.Equal(MaxMessageCount, list.Count);
        Assert.DoesNotContain(list, m => m.Content == "Msg0");
        Assert.Contains(list, m => m.Content == "NewOne");
    }

    [Fact]
    public async Task DeleteAllAsync_Removes_All_Messages()
    {
        await _repository.InsertAsync(NewMessage(ChatRole.User, "A"));
        await _repository.InsertAsync(NewMessage(ChatRole.Assistant, "B"));
        await _repository.DeleteAllAsync();
        var list = await _repository.GetMessagesAsync(10);
        Assert.Empty(list);
    }

    private static DbChatMessage NewMessage(ChatRole role, string content, DateTime? createdAt = null)
    {
        return new DbChatMessage
        {
            Role = role,
            Content = content,
            CreatedAt = createdAt ?? DateTime.UtcNow
        };
    }
}
