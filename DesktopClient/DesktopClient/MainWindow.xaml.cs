using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using DesktopClient.ViewModels;

namespace DesktopClient;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainViewModel(App.ChatOptions, App.ApiOptions);
        Loaded += async (_, _) =>
        {
            if (DataContext is MainViewModel vm)
            {
                vm.Messages.CollectionChanged += (_, _) => Dispatcher.InvokeAsync(ScrollToBottom);
                await vm.LoadConversationAsync();
            }
        };
    }

    private void ScrollToBottom()
    {
        // Keep newest message visible at the bottom.
        if (VisualTreeHelper.GetChild(this, 0) is not DependencyObject root)
            return;

        var scrollViewer = FindDescendant<ScrollViewer>(root);
        scrollViewer?.ScrollToEnd();
    }

    private static T? FindDescendant<T>(DependencyObject root) where T : DependencyObject
    {
        var count = VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < count; i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is T typed)
                return typed;
            var result = FindDescendant<T>(child);
            if (result != null)
                return result;
        }
        return null;
    }
}