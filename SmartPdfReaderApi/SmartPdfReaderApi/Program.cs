using Data.DataContext;
using Data.Repository;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");
if (string.IsNullOrEmpty(connectionString))
{
    throw new InvalidOperationException("Connection string 'DefaultConnection' not found.");
}
builder.Services.AddDbContext<ChatHistoryDbContext>(options =>
    options.UseSqlServer(connectionString, sqlOptions =>
        sqlOptions.EnableRetryOnFailure(
            maxRetryCount: 5,
            maxRetryDelay: TimeSpan.FromSeconds(10),
            errorNumbersToAdd: null)));

var maxMessageCount = builder.Configuration.GetValue("ChatHistory:MaxMessageCount", 1000);
builder.Services.AddScoped<Repository>(sp => new Repository(sp.GetRequiredService<ChatHistoryDbContext>(), maxMessageCount));

var app = builder.Build();

app.Run();

