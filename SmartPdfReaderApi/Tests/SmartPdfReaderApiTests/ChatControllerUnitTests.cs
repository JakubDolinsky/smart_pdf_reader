using Data.Models;
using Data.Repository;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Moq;
using Service.Clients;
using Service.Configuration;
using Service.Models;
using Service.Services;
using SmartPdfReaderApi.Controllers;
using SmartPdfReaderApi.Models;
using Xunit;

namespace SmartPdfReaderApiTests;

/// <summary>
/// Unit tests for <see cref="ChatController"/> with real <see cref="ChatMessageService"/> and mocked <see cref="IRepository"/> and <see cref="IFastApiClient"/>.
/// </summary>
public class ChatControllerUnitTests
{
    private static IOptions<ChatServiceOptions> CreateOptions(int minLength = 1, int maxLength = 2000)
    {
        return Options.Create(new ChatServiceOptions
        {
            MinQuestionLength = minLength,
            MaxQuestionLength = maxLength
        });
    }

    [Fact]
    public async Task AskQuestion_Returns_200_And_Answer_When_Valid()
    {
        var insertCount = 0;
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.InsertAsync(It.IsAny<DbChatMessage>(), It.IsAny<CancellationToken>()))
            .Callback<DbChatMessage, CancellationToken>((m, _) => { insertCount++; m.Id = insertCount; })
            .Returns(Task.CompletedTask);
        mockRepo.Setup(r => r.GetMessagesAsync(5, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<DbChatMessage>());

        var mockClient = new Mock<IFastApiClient>();
        mockClient.Setup(c => c.GetAnswerAsync(It.IsAny<string>(), It.IsAny<IReadOnlyList<BusinessChatMessage>>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync("Hello back");

        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, CreateOptions(), NullLogger<ChatController>.Instance);

        var request = new AskQuestionRequest { Role = ChatRole.User, Content = "Hello", CreatedAt = DateTime.UtcNow };
        var result = await controller.AskQuestion(request, CancellationToken.None);

        var okResult = Assert.IsType<OkObjectResult>(result.Result);
        var response = Assert.IsType<ChatMessageResponse>(okResult.Value);
        Assert.Equal(2, response.Id);
        Assert.Equal(ChatRole.Assistant, response.Role);
        Assert.Equal("Hello back", response.Content);
    }

    [Fact]
    public async Task AskQuestion_Returns_400_When_Request_Is_Null()
    {
        var mockRepo = new Mock<IRepository>();
        var mockClient = new Mock<IFastApiClient>();
        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, CreateOptions(), NullLogger<ChatController>.Instance);

        var result = await controller.AskQuestion(null!, CancellationToken.None);

        Assert.IsType<BadRequestObjectResult>(result.Result);
    }

    [Fact]
    public async Task AskQuestion_Returns_400_When_Content_Below_MinLength()
    {
        var options = CreateOptions(minLength: 5, maxLength: 2000);
        var mockRepo = new Mock<IRepository>();
        var mockClient = new Mock<IFastApiClient>();
        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, options, NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, options, NullLogger<ChatController>.Instance);
        var request = new AskQuestionRequest { Role = ChatRole.User, Content = "Hi" };

        var result = await controller.AskQuestion(request, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result.Result);
        Assert.Contains("5", badRequest.Value?.ToString() ?? "");
    }

    [Fact]
    public async Task AskQuestion_Returns_400_When_Content_Exceeds_MaxLength()
    {
        var options = CreateOptions(minLength: 1, maxLength: 10);
        var mockRepo = new Mock<IRepository>();
        var mockClient = new Mock<IFastApiClient>();
        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, options, NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, options, NullLogger<ChatController>.Instance);
        var request = new AskQuestionRequest { Role = ChatRole.User, Content = "This is too long" };

        var result = await controller.AskQuestion(request, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result.Result);
        Assert.Contains("10", badRequest.Value?.ToString() ?? "");
    }

    [Fact]
    public async Task AskQuestion_Returns_503_When_Service_Throws_InvalidOperationException()
    {
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.InsertAsync(It.IsAny<DbChatMessage>(), It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);
        mockRepo.Setup(r => r.GetMessagesAsync(5, It.IsAny<CancellationToken>())).ReturnsAsync(new List<DbChatMessage>());

        var mockClient = new Mock<IFastApiClient>();
        mockClient.Setup(c => c.GetAnswerAsync(It.IsAny<string>(), It.IsAny<IReadOnlyList<BusinessChatMessage>>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync((string?)null!);

        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, CreateOptions(), NullLogger<ChatController>.Instance);
        var request = new AskQuestionRequest { Role = ChatRole.User, Content = "Hello" };

        var result = await controller.AskQuestion(request, CancellationToken.None);

        var statusResult = Assert.IsType<ObjectResult>(result.Result);
        Assert.Equal(503, statusResult.StatusCode);
    }

    [Fact]
    public async Task LoadAllMessages_Returns_200_And_List()
    {
        var messages = new List<BusinessChatMessage>
        {
            new() { Id = 1, Role = ChatRole.User, Content = "Hi", CreatedAt = DateTime.UtcNow },
            new() { Id = 2, Role = ChatRole.Assistant, Content = "Hello", CreatedAt = DateTime.UtcNow }
        };
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.GetAllMessagesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(
            messages.Select(m => new DbChatMessage { Id = m.Id, Role = m.Role, Content = m.Content, CreatedAt = m.CreatedAt }).ToList());

        var mockClient = new Mock<IFastApiClient>();
        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, CreateOptions(), NullLogger<ChatController>.Instance);

        var result = await controller.LoadAllMessages(CancellationToken.None);

        var okResult = Assert.IsType<OkObjectResult>(result.Result);
        var list = Assert.IsAssignableFrom<IReadOnlyList<ChatMessageResponse>>(okResult.Value);
        Assert.Equal(2, list.Count);
        Assert.Equal("Hi", list[0].Content);
        Assert.Equal("Hello", list[1].Content);
    }

    [Fact]
    public async Task LoadAllMessages_Returns_Empty_List_When_No_Messages()
    {
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.GetAllMessagesAsync(It.IsAny<CancellationToken>())).ReturnsAsync(new List<DbChatMessage>());
        var mockClient = new Mock<IFastApiClient>();
        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, CreateOptions(), NullLogger<ChatController>.Instance);

        var result = await controller.LoadAllMessages(CancellationToken.None);

        var okResult = Assert.IsType<OkObjectResult>(result.Result);
        var list = Assert.IsAssignableFrom<IReadOnlyList<ChatMessageResponse>>(okResult.Value);
        Assert.Empty(list);
    }

    [Fact]
    public async Task DeleteAll_Returns_204_And_Calls_Service()
    {
        var mockRepo = new Mock<IRepository>();
        mockRepo.Setup(r => r.DeleteAllAsync(It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);
        var mockClient = new Mock<IFastApiClient>();
        var service = new ChatMessageService(mockRepo.Object, mockClient.Object, CreateOptions(), NullLogger<ChatMessageService>.Instance);
        var controller = new ChatController(service, CreateOptions(), NullLogger<ChatController>.Instance);

        var result = await controller.DeleteAll(CancellationToken.None);

        Assert.IsType<NoContentResult>(result);
        mockRepo.Verify(r => r.DeleteAllAsync(It.IsAny<CancellationToken>()), Times.Once);
    }
}
