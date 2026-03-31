using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;
using Microsoft.Extensions.Configuration;

namespace Data.DataContext
{
    /// <summary>
    /// Factory for creating <see cref="ChatHistoryDbContext"/> instances, e.g. for EF Core design-time tools (migrations).
    /// </summary>
    public class ChatHistoryDbContextFactory : IDesignTimeDbContextFactory<ChatHistoryDbContext>
    {
        /// <inheritdoc />
        public ChatHistoryDbContext CreateDbContext(string[] args)
        {
            var basePath = GetAppSettingsBasePath();
            var configuration = new ConfigurationBuilder()
                .SetBasePath(basePath)
                .AddJsonFile("appsettings.json", optional: false)
                .AddJsonFile("appsettings.Development.json", optional: true)
                .Build();

            var connectionString = configuration.GetConnectionString("DefaultConnection");
            if (string.IsNullOrEmpty(connectionString))
            {
                throw new InvalidOperationException(
                    "Connection string 'DefaultConnection' not found. " +
                    "Ensure appsettings.json (or environment) is available and contains ConnectionStrings:DefaultConnection.");
            }

            var optionsBuilder = new DbContextOptionsBuilder<ChatHistoryDbContext>();
            optionsBuilder.UseSqlServer(connectionString, sqlOptions =>
                sqlOptions.EnableRetryOnFailure(
                    maxRetryCount: 5,
                    maxRetryDelay: TimeSpan.FromSeconds(10),
                    errorNumbersToAdd: null));

            return new ChatHistoryDbContext(optionsBuilder.Options);
        }

        /// <summary>
        /// Resolves the directory containing appsettings.json (e.g. when run from solution root by the migration script).
        /// </summary>
        private static string GetAppSettingsBasePath()
        {
            var current = Directory.GetCurrentDirectory();
            var candidates = new[]
            {
                current,
                Path.Combine(current, "SmartPdfReaderApi", "SmartPdfReaderApi"),
                Path.Combine(current, "SmartPdfReaderApi"),
            };
            foreach (var dir in candidates)
            {
                if (File.Exists(Path.Combine(dir, "appsettings.json")))
                    return dir;
            }
            return current;
        }
    }
}
