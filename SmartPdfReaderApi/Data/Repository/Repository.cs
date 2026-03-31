using Data.DataContext;
using Data.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Data.Repository
{
    /// <summary>
    /// Provides access to chat history in the database via <see cref="ChatHistoryDbContext"/>.
    /// Supports inserting messages (with automatic removal of oldest when at capacity),
    /// loading a configurable number of recent messages, and clearing all messages.
    /// </summary>
    public class Repository : IRepository
    {
        private readonly ChatHistoryDbContext _context;
        private readonly int _maxMessageCount;
        private readonly ILogger<Repository> _logger;

        public Repository(ChatHistoryDbContext context, int maxMessageCount, ILogger<Repository> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _maxMessageCount = maxMessageCount;
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Inserts one chat message. If the number of stored messages already equals the maximum allowed,
        /// the oldest message is deleted first, then the new message is inserted.
        /// </summary>
        public async Task InsertAsync(DbChatMessage message, CancellationToken cancellationToken = default)
        {
            if (message == null)
                throw new ArgumentNullException(nameof(message));

            var currentCount = await _context.ChatHistory.CountAsync(cancellationToken).ConfigureAwait(false);
            if (currentCount >= _maxMessageCount)
            {
                var oldest = await _context.ChatHistory
                    .OrderBy(m => m.CreatedAt)
                    .FirstOrDefaultAsync(cancellationToken)
                    .ConfigureAwait(false);
                if (oldest != null)
                {
                    _logger.LogDebug("At capacity ({Count}/{Max}), removing oldest message Id={Id}", currentCount, _maxMessageCount, oldest.Id);
                    _context.ChatHistory.Remove(oldest);
                    await _context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
                }
            }

            await _context.ChatHistory.AddAsync(message, cancellationToken).ConfigureAwait(false);
            await _context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogDebug("Inserted message Id={Id}, Role={Role}", message.Id, message.Role);
        }

        /// <summary>
        /// Loads the most recent messages from the database, in chronological order (oldest first).
        /// Used for initial chat history load and for loading the last N messages (e.g. last 4 for RAG).
        /// </summary>
        /// <param name="count">Maximum number of messages to return (most recent).</param>
        /// <returns>Messages ordered by CreatedAt ascending (oldest to newest).</returns>
        public async Task<IReadOnlyList<DbChatMessage>> GetMessagesAsync(int count, CancellationToken cancellationToken = default)
        {
            if (count <= 0)
                return Array.Empty<DbChatMessage>();

            var messages = await _context.ChatHistory
                .OrderByDescending(m => m.CreatedAt)
                .Take(count)
                .ToListAsync(cancellationToken)
                .ConfigureAwait(false);

            messages.Reverse();
            _logger.LogDebug("GetMessagesAsync requested {Count}, returning {Returned} messages", count, messages.Count);
            return messages;
        }

        /// <summary>
        /// Loads all messages from the database in chronological order (oldest first).
        /// </summary>
        public async Task<IReadOnlyList<DbChatMessage>> GetAllMessagesAsync(CancellationToken cancellationToken = default)
        {
            var messages = await _context.ChatHistory
                .OrderBy(m => m.CreatedAt)
                .ToListAsync(cancellationToken)
                .ConfigureAwait(false);
            _logger.LogDebug("GetAllMessagesAsync returning {Count} messages", messages.Count);
            return messages;
        }

        /// <summary>
        /// Deletes all messages from the database.
        /// </summary>
        public async Task DeleteAllAsync(CancellationToken cancellationToken = default)
        {
            var all = await _context.ChatHistory.ToListAsync(cancellationToken).ConfigureAwait(false);
            _context.ChatHistory.RemoveRange(all);
            await _context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("DeleteAllAsync removed {Count} messages", all.Count);
        }
    }
}
