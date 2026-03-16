using System.Net;
using System.Text;
using System.Text.Json;

namespace ServiceTests;

/// <summary>
/// Returns a fixed JSON answer for POST /ask to simulate the RAG FastAPI.
/// </summary>
internal sealed class FakeRagResponseHandler : HttpMessageHandler
{
    private readonly string _answer;

    public FakeRagResponseHandler(string answer = "Four")
    {
        _answer = answer;
    }

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(new { answer = _answer });
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json")
        });
    }
}
