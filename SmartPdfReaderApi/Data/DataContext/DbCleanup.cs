using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Data.DataContext
{
    /// <summary>
    /// Cleans the database (e.g. after integration tests) by removing all chat history.
    /// </summary>
    public class DbCleanup
    {
        private readonly ChatHistoryDbContext _context;
        private readonly ILogger<DbCleanup> _logger;

        public DbCleanup(ChatHistoryDbContext context, ILogger<DbCleanup> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Deletes all messages from the ChatHistory table.
        /// </summary>
        public async Task CleanAsync(CancellationToken cancellationToken = default)
        {
            var all = await _context.ChatHistory.ToListAsync(cancellationToken).ConfigureAwait(false);
            _context.ChatHistory.RemoveRange(all);
            await _context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("CleanAsync removed {Count} messages from ChatHistory", all.Count);
        }
    }
}
