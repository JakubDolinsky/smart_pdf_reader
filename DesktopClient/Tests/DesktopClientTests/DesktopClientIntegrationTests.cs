using System.Net.Http;
using DesktopClient.Client;
using DesktopClient.Configuration;
using DesktopClient.ModelsDTO;
using Microsoft.Extensions.Configuration;
using ClientChatRole = DesktopClient.Models.ChatRole;
using ApiChatRole = DesktopClient.ModelsDTO.ChatRole;

namespace DesktopClientTests;

/// <summary>
/// Integration tests (formerly in DesktopClientIntegrationTests project).
/// Requires SmartPdfReaderApi (and RAG when asking questions) to be running.
/// Any Web API or RAG failure is reported as a test error.
/// BaseUrl and TimeoutSeconds are read from this project's appsettings.json (keep TimeoutSeconds in sync with SmartPdfReaderApi/appsettings.json "DesktopClient": "TimeoutSeconds").
/// </summary>
public sealed class DesktopClientIntegrationTests
{
    private static readonly Lazy<ApiOptions> LazyApiOptions = new(() =>
    {
        var config = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
            .Build();
        var options = new ApiOptions();
        config.GetSection(ApiOptions.SectionName).Bind(options);
        return options;
    });

    private static string GetApiBaseUrl()
    {
        var url = LazyApiOptions.Value.BaseUrl;
        if (!string.IsNullOrWhiteSpace(url)) return url.Trim();
        url = Environment.GetEnvironmentVariable("SMART_PDF_READER_API_BASE_URL");
        return !string.IsNullOrWhiteSpace(url) ? url.Trim() : "http://localhost:5000";
    }

    private static HttpClient CreateHttpClient()
    {
        return new HttpClient { Timeout = TimeSpan.FromSeconds(LazyApiOptions.Value.TimeoutSeconds) };
    }

    [Fact]
    public async Task LoadAllMessages_ReturnsNotNull()
    {
        var baseUrl = GetApiBaseUrl();
        var client = new SmartPdfReaderApiClient(baseUrl, CreateHttpClient());

        // Throws on HTTP/server error (e.g. API down, 500) -> test fails with clear error
        var messages = await client.LoadAllMessagesAsync();

        Assert.NotNull(messages);
    }

    [Fact]
    public async Task AskQuestion_ReturnsAssistantResponse()
    {
        var baseUrl = GetApiBaseUrl();
        var client = new SmartPdfReaderApiClient(baseUrl, CreateHttpClient());

        var request = ChatDtoMapper.ToAskQuestionRequest(
            content: "Integration test: say hi",
            clientRole: ClientChatRole.User);

        // Throws on HTTP/server/RAG error (e.g. 503 from RAG) -> test fails with clear error
        var response = await client.AskQuestionAsync(request);

        Assert.NotNull(response);
        Assert.False(string.IsNullOrWhiteSpace(response.Content), "Assistant response content should not be null or empty (RAG/API may have failed).");
        Assert.Equal(ApiChatRole._1, response.Role); // API enum ordering: User=0, Assistant=1
    }

    /// <summary>
    /// Deletes the whole conversation via the API. Verifies delete and cleans up DB after integration tests.
    /// </summary>
    [Fact]
    public async Task DeleteAll_ClearsConversation()
    {
        var baseUrl = GetApiBaseUrl();
        var client = new SmartPdfReaderApiClient(baseUrl, CreateHttpClient());

        await client.DeleteAllAsync();

        var messages = await client.LoadAllMessagesAsync();
        Assert.NotNull(messages);
        Assert.Empty(messages);
    }
}
