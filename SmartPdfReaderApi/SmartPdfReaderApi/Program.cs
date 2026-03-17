using Data.DataContext;
using Data.Repository;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Serilog;
using Service.Clients;
using Service.Configuration;
using Service.Services;

var builder = WebApplication.CreateBuilder(args);

var solutionRoot = Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, ".."));
var fileLogEnabled = builder.Configuration.GetValue("Logging:File:Enabled", false);
var fileLogName = builder.Configuration.GetValue("Logging:File:FileName", "SmartPdfReaderApi.log");
var logPath = Path.Combine(solutionRoot, "logs", fileLogName);

var loggerConfig = new LoggerConfiguration()
    .ReadFrom.Configuration(builder.Configuration)
    .Enrich.FromLogContext()
    .WriteTo.Console();

if (fileLogEnabled)
    loggerConfig = loggerConfig.WriteTo.File(logPath, rollingInterval: RollingInterval.Day, retainedFileCountLimit: 7, shared: true);

Log.Logger = loggerConfig.CreateLogger();
builder.Host.UseSerilog();

if (fileLogEnabled)
    Log.Information("File logging enabled. Log path: {LogPath}", logPath);
else
    Log.Debug("File logging disabled (Logging:File:Enabled = false).");

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

var maxMessageCount = builder.Configuration.GetValue("ChatHistory:MaxMessageCount", 100);
builder.Services.AddScoped<Repository>(sp => new Repository(sp.GetRequiredService<ChatHistoryDbContext>(), maxMessageCount, sp.GetRequiredService<ILogger<Repository>>()));
builder.Services.AddScoped<IRepository>(sp => sp.GetRequiredService<Repository>());

builder.Services.Configure<ChatServiceOptions>(
    builder.Configuration.GetSection(ChatServiceOptions.SectionName));

builder.Services.AddHttpClient<FastApiClient>((sp, client) =>
{
    var options = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<ChatServiceOptions>>().Value;
    client.BaseAddress = new Uri(options.FastApiBaseUrl.TrimEnd('/') + "/");
    client.Timeout = TimeSpan.FromSeconds(Math.Max(60, options.FastApiTimeoutSeconds));
});
builder.Services.AddScoped<IFastApiClient>(sp => sp.GetRequiredService<FastApiClient>());
builder.Services.AddScoped<ChatMessageService>();

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "Smart PDF Reader API",
        Version = "v1",
        Description = "API for chat: ask questions (RAG), load messages, clear conversation."
    });
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(options =>
    {
        options.SwaggerEndpoint("/swagger/v1/swagger.json", "Smart PDF Reader API v1");
    });
}

app.MapControllers();

try
{
    Log.Information("Application starting.");
    app.Run();
}
finally
{
    Log.CloseAndFlush();
}

