using Service.Models;

namespace Service.Clients;

/// <summary>
/// Client for the RAG FastAPI: sends question + last 4 messages and returns the answer string.
/// </summary>
public interface IFastApiClient
{
    /// <summary>
    /// Sends the question and last 4 previous messages to the RAG FastAPI and returns the answer string.
    /// </summary>
    /// <param name="question">Current user question.</param>
    /// <param name="lastFourMessages">Up to 4 previous messages (oldest first).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The answer string from the FastAPI.</returns>
    Task<string> GetAnswerAsync(
        string question,
        IReadOnlyList<BusinessChatMessage> lastFourMessages,
        CancellationToken cancellationToken = default);
}
