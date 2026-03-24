using System;
using System.IO;
using System.Windows;
using Microsoft.Extensions.Configuration;
using DesktopClient.Configuration;
using Serilog;
using Serilog.Events;

namespace DesktopClient;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    public static ChatOptions ChatOptions { get; private set; } = null!;
    public static ApiOptions ApiOptions { get; private set; } = null!;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        try
        {
            ConfigureSerilog();
            Log.Information("DesktopClient starting. Environment: {Environment}", Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT"));

            Current.DispatcherUnhandledException += (_, args) =>
            {
                Log.Error(args.Exception, "Unhandled UI exception.");
                MessageBox.Show("The application encountered an unexpected error. Please see logs for details.", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                args.Handled = true;
            };

            AppDomain.CurrentDomain.UnhandledException += (_, args) =>
            {
                var ex = args.ExceptionObject as Exception;
                Log.Error(ex, "Unhandled non-UI exception.");
                try { MessageBox.Show(ex?.ToString() ?? "Unknown error", "Fatal Error", MessageBoxButton.OK, MessageBoxImage.Error); } catch { }
            };

            // Use executable directory so config is found when debugger sets working directory to solution/project folder.
            var config = new ConfigurationBuilder()
                .SetBasePath(AppContext.BaseDirectory)
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
                .Build();

            ChatOptions = new ChatOptions();
            config.GetSection(ChatOptions.SectionName).Bind(ChatOptions);

            ApiOptions = new ApiOptions();
            config.GetSection(ApiOptions.SectionName).Bind(ApiOptions);

            // Create and show main window explicitly so any failure is caught below.
            var mainWindow = new MainWindow();
            mainWindow.Show();
        }
        catch (Exception ex)
        {
            Log.Error(ex, "Startup failed.");
            MessageBox.Show(
                ex.ToString(),
                "Startup Error – cannot show window",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            Shutdown(1);
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        try
        {
            Log.Information("DesktopClient shutting down.");
        }
        finally
        {
            Log.CloseAndFlush();
        }
        base.OnExit(e);
    }

    private static void ConfigureSerilog()
    {
        try
        {
            // DesktopClient is located at: .../DesktopClient/DesktopClient/
            // Output is:            .../DesktopClient/DesktopClient/bin/Debug/netX.Y-windows/
            // So we go 4 levels up to reach the project folder.
            var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
            var logsDir = Path.Combine(projectRoot, "Logs");
            Directory.CreateDirectory(logsDir);

            var logPath = Path.Combine(logsDir, "DesktopClient.log");

            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Override("Microsoft", LogEventLevel.Warning)
                .MinimumLevel.Information()
                .WriteTo.File(
                    logPath,
                    rollingInterval: RollingInterval.Day,
                    retainedFileCountLimit: 14,
                    shared: true)
                .CreateLogger();
        }
        catch
        {
            // Avoid crashing on logging setup; fall back to default logger.
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .CreateLogger();
        }
    }
}

