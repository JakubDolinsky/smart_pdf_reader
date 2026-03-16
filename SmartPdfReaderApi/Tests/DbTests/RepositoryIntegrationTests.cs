using Data.DataContext;
using Data.Models;
using Data.Repository;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace DbTests;

/// <summary>
/// Integration tests against a real SQL Server database.
/// Requires a valid connection string in appsettings.json or environment.
/// Uses <see cref="DbCleanup"/> after tests to leave the database clean.
/// </summary>
public class RepositoryIntegrationTests : IAsyncLifetime
{
    private ChatHistoryDbContext _context = null!;
    private Repository _repository = null!;
    private DbCleanup _cleanup = null!;
    private const int MaxMessageCount = 10;

    public async Task InitializeAsync()
    {
        var connectionString = GetConnectionString();
        if (string.IsNullOrEmpty(connectionString))
        {
            throw new InvalidOperationException(
                "Integration tests require ConnectionStrings:DefaultConnection in appsettings.json or environment.");
        }

        var options = new DbContextOptionsBuilder<ChatHistoryDbContext>()
            .UseSqlServer(connectionString)
            .Options;
        _context = new ChatHistoryDbContext(options);
        await _context.Database.CanConnectAsync();
        _repository = new Repository(_context, MaxMessageCount, NullLogger<Repository>.Instance);
        _cleanup = new DbCleanup(_context, NullLogger<DbCleanup>.Instance);
        await _cleanup.CleanAsync();
    }

    public async Task DisposeAsync() => await _cleanup.CleanAsync();

    private static string? GetConnectionString()
    {
        var config = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true)
            .Build();
        return config.GetConnectionString("DefaultConnection");
    }

    [Fact]
    public async Task Database_Connection_Succeeds()
    {
        var canConnect = await _context.Database.CanConnectAsync();
        Assert.True(canConnect);
    }

    [Fact]
    public async Task Insert_And_GetMessages_Against_Real_Db()
    {
        await _repository.InsertAsync(NewMessage(ChatRole.User, "Integration user message"));
        await _repository.InsertAsync(NewMessage(ChatRole.Assistant, "Integration assistant reply"));
        var list = await _repository.GetMessagesAsync(10);
        Assert.Equal(2, list.Count);
        Assert.Equal("Integration user message", list[0].Content);
        Assert.Equal("Integration assistant reply", list[1].Content);
    }

    [Fact]
    public async Task GetMessages_Last_Four_For_RAG_Scenario()
    {
        for (int i = 1; i <= 6; i++)
            await _repository.InsertAsync(NewMessage(i % 2 == 1 ? ChatRole.User : ChatRole.Assistant, $"Msg{i}"));
        var lastFour = await _repository.GetMessagesAsync(4);
        Assert.Equal(4, lastFour.Count);
        Assert.Equal("Msg3", lastFour[0].Content);
        Assert.Equal("Msg4", lastFour[1].Content);
        Assert.Equal("Msg5", lastFour[2].Content);
        Assert.Equal("Msg6", lastFour[3].Content);
    }

    [Fact]
    public async Task Insert_Evicts_Oldest_When_At_Capacity_Real_Db()
    {
        for (int i = 0; i < MaxMessageCount; i++)
            await _repository.InsertAsync(NewMessage(ChatRole.User, $"Evict{i}"));
        await _repository.InsertAsync(NewMessage(ChatRole.Assistant, "NewAfterEvict"));
        var list = await _repository.GetMessagesAsync(MaxMessageCount + 1);
        Assert.Equal(MaxMessageCount, list.Count);
        Assert.DoesNotContain(list, m => m.Content == "Evict0");
        Assert.Contains(list, m => m.Content == "NewAfterEvict");
    }

    [Fact]
    public async Task DeleteAll_Clears_Table_Real_Db()
    {
        await _repository.InsertAsync(NewMessage(ChatRole.User, "ToDelete"));
        await _repository.DeleteAllAsync();
        var list = await _repository.GetMessagesAsync(10);
        Assert.Empty(list);
    }

    private static DbChatMessage NewMessage(ChatRole role, string content)
    {
        return new DbChatMessage
        {
            Role = role,
            Content = content,
            CreatedAt = DateTime.UtcNow
        };
    }
}
