using Data.Models;
using Data.Repository;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Moq;
using Service.Configuration;
using Service.Clients;
using Service.Models;
using Service.Services;
using Xunit;

namespace ServiceTests;

/// <summary>
/// Unit tests for <see cref="ChatMessageService"/> with mocked <see cref="IRepository"/> and <see cref="IFastApiClient"/>.
/// </summary>
public class ChatMessageServiceUnitTests
{
    private static IOptions<ChatServiceOptions> CreateOptions(int maxQuestionLength = 2000)
    {
        return Microsoft.Extensions.Options.Options.Create(new ChatServiceOptions { MaxQuestionLength = maxQuestionLength });
    }

    [Fact]
    public void Constructor_Throws_When_Repository_Is_Null()
    {
        var client = new Mock<IFastApiClient>().Object;
        Assert.Throws<ArgumentNullException>(() =>
            new ChatMessageService(null!, client, CreateOptions(), NullLogger<ChatMessageService>.Instance));
    }

    [Fact]
    public void Constructor_Throws_When_IFastApiClient_Is_Null()
    {
        var repo = new Mock<IRepository>().Object;
        Assert.Throws<ArgumentNullException>(() =>
            new ChatMessageService(repo, null!, CreateOptions(), NullLogger<ChatMessageService>.Instance));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_Throws_When_MaxQuestionLength_Not_Positive(int maxLen)
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var options = Microsoft.Extensions.Options.Options.Create(new ChatServiceOptions { MaxQuestionLength = maxLen });
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ChatMessageService(repo, client, options, NullLogger<ChatMessageService>.Instance));
    }

    [Fact]
    public void ValidateQuestionLength_Throws_When_Question_Is_Null()
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var service = new ChatMessageService(repo, client, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        Assert.Throws<ArgumentNullException>(() => service.ValidateQuestionLength(null!));
    }

    [Fact]
    public void ValidateQuestionLength_Throws_When_Content_Exceeds_Max()
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var options = CreateOptions(10);
        var service = new ChatMessageService(repo, client, options, NullLogger<ChatMessageService>.Instance);
        var question = new BusinessChatMessage { Content = new string('x', 11) };
        var ex = Assert.Throws<ArgumentException>(() => service.ValidateQuestionLength(question));
        Assert.Contains("11", ex.Message);
        Assert.Contains("10", ex.Message);
    }

    [Fact]
    public void ValidateQuestionLength_Throws_When_Content_Is_Empty()
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var service = new ChatMessageService(repo, client, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var question = new BusinessChatMessage { Content = string.Empty };
        Assert.Throws<ArgumentException>(() => service.ValidateQuestionLength(question));
    }

    [Fact]
    public void ValidateQuestionLength_Throws_When_Content_Is_Null()
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var service = new ChatMessageService(repo, client, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var question = new BusinessChatMessage { Content = null! };
        Assert.Throws<ArgumentException>(() => service.ValidateQuestionLength(question));
    }

    [Fact]
    public void ValidateQuestionLength_Does_Not_Throw_When_At_Max_Length()
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var options = CreateOptions(10);
        var service = new ChatMessageService(repo, client, options, NullLogger<ChatMessageService>.Instance);
        var question = new BusinessChatMessage { Content = new string('x', 10) };
        service.ValidateQuestionLength(question);
    }

    [Fact]
    public async Task AskAsync_Throws_When_Question_Is_Null()
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var service = new ChatMessageService(repo, client, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        await Assert.ThrowsAsync<ArgumentNullException>(() => service.AskAsync(null!));
    }

    [Fact]
    public async Task AskAsync_Throws_When_Question_Too_Long()
    {
        var repo = new Mock<IRepository>().Object;
        var client = new Mock<IFastApiClient>().Object;
        var options = CreateOptions(5);
        var service = new ChatMessageService(repo, client, options, NullLogger<ChatMessageService>.Instance);
        var question = new BusinessChatMessage { Content = "hello world" };
        await Assert.ThrowsAsync<ArgumentException>(() => service.AskAsync(question));
    }

    [Fact]
    public async Task AskAsync_Throws_When_Answer_Is_Empty()
    {
        var repo = new Mock<IRepository>();
        repo.Setup(r => r.InsertAsync(It.IsAny<DbChatMessage>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);
        repo.Setup(r => r.GetMessagesAsync(5, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<DbChatMessage>());

        var client = new Mock<IFastApiClient>();
        client.Setup(c => c.GetAnswerAsync(It.IsAny<string>(), It.IsAny<IReadOnlyList<BusinessChatMessage>>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(string.Empty);

        var service = new ChatMessageService(repo.Object, client.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var question = new BusinessChatMessage { Content = "Hi" };

        await Assert.ThrowsAsync<InvalidOperationException>(() => service.AskAsync(question));
    }

    [Fact]
    public async Task AskAsync_Throws_When_Answer_Is_Null()
    {
        var repo = new Mock<IRepository>();
        repo.Setup(r => r.InsertAsync(It.IsAny<DbChatMessage>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);
        repo.Setup(r => r.GetMessagesAsync(5, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<DbChatMessage>());

        var client = new Mock<IFastApiClient>();
        client.Setup(c => c.GetAnswerAsync(It.IsAny<string>(), It.IsAny<IReadOnlyList<BusinessChatMessage>>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync((string?)null!);

        var service = new ChatMessageService(repo.Object, client.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var question = new BusinessChatMessage { Content = "Hi" };

        await Assert.ThrowsAsync<InvalidOperationException>(() => service.AskAsync(question));
    }

    [Fact]
    public async Task AskAsync_Inserts_Question_Then_Loads_Five_Then_Calls_Client_And_Inserts_Answer()
    {
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.InsertAsync(It.IsAny<DbChatMessage>(), It.IsAny<CancellationToken>()))
            .Callback<DbChatMessage, CancellationToken>((m, _) => { if (m.Id == 0) m.Id = 1; else m.Id = 2; })
            .Returns(Task.CompletedTask);

        var questionAsDb = new DbChatMessage
        {
            Id = 1,
            Role = ChatRole.User,
            Content = "What is 2+2?",
            CreatedAt = DateTime.UtcNow
        };
        mockRepo.Setup(r => r.GetMessagesAsync(5, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<DbChatMessage> { questionAsDb });

        var httpClient = new HttpClient(new FakeRagResponseHandler("Four")) { BaseAddress = new Uri("http://localhost/") };
        IFastApiClient fastApiClient = new FastApiClient(httpClient, NullLogger<FastApiClient>.Instance);

        var service = new ChatMessageService(mockRepo.Object, fastApiClient, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var questionMsg = new BusinessChatMessage
        {
            Content = "What is 2+2?",
            Role = ChatRole.User,
            CreatedAt = DateTime.UtcNow
        };
        var result = await service.AskAsync(questionMsg);

        mockRepo.Verify(r => r.InsertAsync(It.IsAny<DbChatMessage>(), It.IsAny<CancellationToken>()), Times.Exactly(2));
        mockRepo.Verify(r => r.GetMessagesAsync(5, It.IsAny<CancellationToken>()), Times.Once);
        Assert.Equal(ChatRole.Assistant, result.Role);
        Assert.Equal("Four", result.Content);
    }

    [Fact]
    public async Task LoadConversationAsync_Calls_GetAllMessagesAsync_And_Converts_To_Business()
    {
        var dbMessages = new List<DbChatMessage>
        {
            new() { Id = 1, Role = ChatRole.User, Content = "Hi", CreatedAt = DateTime.UtcNow },
            new() { Id = 2, Role = ChatRole.Assistant, Content = "Hello", CreatedAt = DateTime.UtcNow }
        };
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.GetAllMessagesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(dbMessages);

        var client = new Mock<IFastApiClient>().Object;
        var service = new ChatMessageService(mockRepo.Object, client, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var result = await service.LoadConversationAsync();

        mockRepo.Verify(r => r.GetAllMessagesAsync(It.IsAny<CancellationToken>()), Times.Once);
        Assert.Equal(2, result.Count);
        Assert.Equal("Hi", result[0].Content);
        Assert.Equal("Hello", result[1].Content);
    }

    [Fact]
    public async Task DeleteAllMessagesAsync_Calls_Repository_DeleteAllAsync()
    {
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.DeleteAllAsync(It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);
        var client = new Mock<IFastApiClient>().Object;
        var service = new ChatMessageService(mockRepo.Object, client, CreateOptions(), NullLogger<ChatMessageService>.Instance);

        await service.DeleteAllMessagesAsync();

        mockRepo.Verify(r => r.DeleteAllAsync(It.IsAny<CancellationToken>()), Times.Once);
    }
}
