using Data.Models;

namespace Data.Repository;

/// <summary>
/// Provides access to chat history (insert, load recent, load all, delete all).
/// </summary>
public interface IRepository
{
    Task InsertAsync(DbChatMessage message, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<DbChatMessage>> GetMessagesAsync(int count, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<DbChatMessage>> GetAllMessagesAsync(CancellationToken cancellationToken = default);
    Task DeleteAllAsync(CancellationToken cancellationToken = default);
}
