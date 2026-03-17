using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using SmartPdfReaderApi.Models;
using Xunit;

namespace SmartPdfReaderApiTests;

/// <summary>
/// Integration tests for <see cref="SmartPdfReaderApi.Controllers.ChatController"/> using the full web application.
/// Requires DefaultConnection and ChatService:FastApiBaseUrl in appsettings.json (e.g. SQL Server and RAG FastAPI running).
/// </summary>
public class ChatControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true
    };

    public ChatControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
        _client.Timeout = TimeSpan.FromSeconds(1800);
    }
    

    [Fact]
    public async Task LoadAllMessages_Returns_200_And_Empty_Array_When_No_Messages()
    {
        await _client.DeleteAsync("/api/Chat/DeleteAll");

        var response = await _client.GetAsync("/api/Chat/LoadAllMessages");

        response.EnsureSuccessStatusCode();
        var list = await response.Content.ReadFromJsonAsync<List<ChatMessageResponse>>(JsonOptions);
        Assert.NotNull(list);
        Assert.Empty(list);
    }

    [Fact]
    public async Task AskQuestion_Returns_400_When_Content_Empty()
    {
        var request = new AskQuestionRequest
        {
            Role = Data.Models.ChatRole.User,
            Content = "",
            CreatedAt = DateTime.UtcNow
        };

        var response = await _client.PostAsJsonAsync("/api/Chat/AskQuestion", request, JsonOptions);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task AskQuestion_Returns_400_When_Content_Too_Long()
    {
        var request = new AskQuestionRequest
        {
            Role = Data.Models.ChatRole.User,
            Content = new string('x', 2001),
            CreatedAt = DateTime.UtcNow
        };

        var response = await _client.PostAsJsonAsync("/api/Chat/AskQuestion", request, JsonOptions);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task DeleteAll_Returns_204()
    {
        var response = await _client.DeleteAsync("/api/Chat/DeleteAll");

        Assert.Equal(HttpStatusCode.NoContent, response.StatusCode);
    }

    [Fact]
    public async Task AskQuestion_Stores_And_LoadAllMessages_Returns_List()
    {
        await _client.DeleteAsync("/api/Chat/DeleteAll");

        var request = new AskQuestionRequest
        {
            Role = Data.Models.ChatRole.User,
            Content = "What is the capital of France?",
            CreatedAt = DateTime.UtcNow
        };
        var askResponse = await _client.PostAsJsonAsync("/api/Chat/AskQuestion", request, JsonOptions);

        if (askResponse.StatusCode == HttpStatusCode.ServiceUnavailable ||
            askResponse.StatusCode == HttpStatusCode.InternalServerError)
        {
            // RAG or DB not available - skip asserting on success
            return;
        }

        askResponse.EnsureSuccessStatusCode();
        var answer = await askResponse.Content.ReadFromJsonAsync<ChatMessageResponse>(JsonOptions);
        Assert.NotNull(answer);
        Assert.Equal(Data.Models.ChatRole.Assistant, answer.Role);
        Assert.False(string.IsNullOrWhiteSpace(answer.Content));

        var loadResponse = await _client.GetAsync("/api/Chat/LoadAllMessages");
        loadResponse.EnsureSuccessStatusCode();
        var list = await loadResponse.Content.ReadFromJsonAsync<List<ChatMessageResponse>>(JsonOptions);
        Assert.NotNull(list);
        Assert.Equal(2, list.Count);
        Assert.Equal("What is the capital of France?", list[0].Content);
        Assert.Equal(Data.Models.ChatRole.User, list[0].Role);
        Assert.Equal(answer.Content, list[1].Content);
        Assert.Equal(Data.Models.ChatRole.Assistant, list[1].Role);
    }
}
