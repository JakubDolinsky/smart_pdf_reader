using Data.DataContext;
using Data.Models;
using Data.Repository;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Service.Clients;
using Service.Configuration;
using Service.Models;
using Service.Services;
using Xunit;

namespace ServiceTests;

/// <summary>
/// Integration tests for <see cref="ChatMessageService"/> with real SQL Server database
/// and real RAG FastAPI. Requires DefaultConnection and ChatService:FastApiBaseUrl in appsettings.json.
/// Cleans the database before and after tests.
/// </summary>
public class ChatMessageServiceIntegrationTests : IAsyncLifetime
{
    private ChatHistoryDbContext _context = null!;
    private IRepository _repository = null!;
    private ChatMessageService _service = null!;
    private DbCleanup _cleanup = null!;
    private const int MaxMessageCount = 20;

    public async Task InitializeAsync()
    {
        var connectionString = GetConfig().GetConnectionString("DefaultConnection");
        if (string.IsNullOrEmpty(connectionString))
        {
            throw new InvalidOperationException(
                "Integration tests require ConnectionStrings:DefaultConnection in appsettings.json.");
        }

        var fastApiBaseUrl = GetConfig().GetSection("ChatService")["FastApiBaseUrl"] ?? "http://localhost:8000";
        if (string.IsNullOrWhiteSpace(fastApiBaseUrl))
        {
            throw new InvalidOperationException(
                "Integration tests require ChatService:FastApiBaseUrl in appsettings.json (e.g. http://localhost:8000).");
        }

        var options = new DbContextOptionsBuilder<ChatHistoryDbContext>()
            .UseSqlServer(connectionString)
            .Options;
        _context = new ChatHistoryDbContext(options);
        await _context.Database.CanConnectAsync();

        _repository = new Repository(_context, MaxMessageCount, NullLogger<Repository>.Instance);
        _cleanup = new DbCleanup(_context, NullLogger<DbCleanup>.Instance);
        await _cleanup.CleanAsync();

        var httpClient = new HttpClient
        {
            BaseAddress = new Uri(fastApiBaseUrl.TrimEnd('/') + "/"),
            Timeout = TimeSpan.FromSeconds(GetConfig().GetValue("ChatService:FastApiTimeoutSeconds", 900))
        };
        var fastApiClient = new FastApiClient(httpClient, NullLogger<FastApiClient>.Instance);
        var chatOptions = Options.Create(new ChatServiceOptions
        {
            MaxQuestionLength = GetConfig().GetValue("ChatService:MaxQuestionLength", 2000),
            FastApiBaseUrl = fastApiBaseUrl
        });
        _service = new ChatMessageService(_repository, fastApiClient, chatOptions, NullLogger<ChatMessageService>.Instance);
    }

    public async Task DisposeAsync() => await _cleanup.CleanAsync();

    private static IConfiguration GetConfig()
    {
        return new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true)
            .Build();
    }

    [Fact]
    public async Task LoadConversationAsync_Returns_Empty_When_No_Messages()
    {
        var result = await _service.LoadConversationAsync();
        Assert.Empty(result);
    }

    [Fact]
    public async Task AskAsync_Stores_Question_And_Answer_And_Returns_Answer_Model()
    {
        var question = new BusinessChatMessage
        {
            Content = "What is the capital of France?",
            Role = ChatRole.User,
            CreatedAt = DateTime.UtcNow
        };

        var answer = await _service.AskAsync(question);

        Assert.Equal(ChatRole.Assistant, answer.Role);
        Assert.False(string.IsNullOrWhiteSpace(answer.Content));
        Assert.True(answer.Id > 0);
        Assert.True(answer.CreatedAt != default);

        var all = await _service.LoadConversationAsync();
        Assert.Equal(2, all.Count);
        Assert.Equal("What is the capital of France?", all[0].Content);
        Assert.Equal(ChatRole.User, all[0].Role);
        Assert.Equal(answer.Content, all[1].Content);
        Assert.Equal(ChatRole.Assistant, all[1].Role);
    }

    [Fact]
    public async Task AskAsync_With_Default_CreatedAt_Still_Stores_And_Returns()
    {
        var question = new BusinessChatMessage
        {
            Content = "Hello",
            Role = ChatRole.User
        };

        var answer = await _service.AskAsync(question);

        Assert.False(string.IsNullOrWhiteSpace(answer.Content));
        var all = await _service.LoadConversationAsync();
        Assert.Equal(2, all.Count);
        Assert.True(all[0].CreatedAt != default);
    }

    [Fact]
    public async Task DeleteAllMessagesAsync_Clears_Conversation()
    {
        var question = new BusinessChatMessage { Content = "Q", Role = ChatRole.User, CreatedAt = DateTime.UtcNow };
        await _service.AskAsync(question);
        var before = await _service.LoadConversationAsync();
        Assert.Equal(2, before.Count);

        await _service.DeleteAllMessagesAsync();

        var after = await _service.LoadConversationAsync();
        Assert.Empty(after);
    }

    [Fact]
    public async Task AskAsync_With_History_Returns_Answer_And_Stores_All_Messages()
    {
        await _service.AskAsync(new BusinessChatMessage
        {
            Content = "First question",
            Role = ChatRole.User,
            CreatedAt = DateTime.UtcNow.AddMinutes(-2)
        });
        await _service.AskAsync(new BusinessChatMessage
        {
            Content = "Second question",
            Role = ChatRole.User,
            CreatedAt = DateTime.UtcNow.AddMinutes(-1)
        });

        var thirdQuestion = new BusinessChatMessage
        {
            Content = "Third question",
            Role = ChatRole.User,
            CreatedAt = DateTime.UtcNow
        };
        var answer = await _service.AskAsync(thirdQuestion);

        Assert.Equal(ChatRole.Assistant, answer.Role);
        Assert.False(string.IsNullOrWhiteSpace(answer.Content));
        var all = await _service.LoadConversationAsync();
        Assert.Equal(6, all.Count);
    }
}
