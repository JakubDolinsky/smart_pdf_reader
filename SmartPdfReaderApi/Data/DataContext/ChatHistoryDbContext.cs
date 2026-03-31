using Data.Models;
using Microsoft.EntityFrameworkCore;

namespace Data.DataContext
{
    /// <summary>
    /// EF Core context for persisting chat history (RAG/conversation messages).
    /// </summary>
    public class ChatHistoryDbContext : DbContext
    {
        public ChatHistoryDbContext(DbContextOptions<ChatHistoryDbContext> options)
            : base(options)
        {
        }

        public DbSet<DbChatMessage> ChatHistory => Set<DbChatMessage>();

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<DbChatMessage>(e =>
            {
                e.ToTable("ChatHistory");
                e.HasKey(x => x.Id);
                e.Property(x => x.Role).HasConversion<string>().HasMaxLength(32).IsRequired();
                e.Property(x => x.Content).IsRequired();
                e.HasIndex(x => x.CreatedAt);
            });
        }
    }
}
