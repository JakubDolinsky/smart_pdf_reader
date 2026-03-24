using ClientChatRole = DesktopClient.Models.ChatRole;
using ApiChatRole = DesktopClient.ModelsDTO.ChatRole;
using DesktopClient.ModelsDTO;

namespace DesktopClientTests;

/// <summary>
/// Unit tests (formerly in DesktopCLientUnitTests project).
/// </summary>
public sealed class DesktopCLientUnitTests
{
    [Fact]
    public void ToChatModel_MapsRoleAndContent_AndTimestampUtc()
    {
        var dto = new ChatMessageResponse
        {
            Id = 123,
            Role = ApiChatRole._0,
            Content = "Hello",
            CreatedAt = new DateTimeOffset(2026, 1, 2, 3, 4, 5, TimeSpan.Zero)
        };

        var model = ChatDtoMapper.ToChatModel(dto);

        Assert.Equal(ClientChatRole.User, model.Role);
        Assert.Equal("Hello", model.Content);
        Assert.Equal(new DateTime(2026, 1, 2, 3, 4, 5, DateTimeKind.Utc), model.Timestamp.ToUniversalTime());
    }

    [Fact]
    public void ToChatModel_MapsAssistantRole()
    {
        var dto = new ChatMessageResponse
        {
            Role = ApiChatRole._1,
            Content = "Answer",
            CreatedAt = DateTimeOffset.UtcNow
        };

        var model = ChatDtoMapper.ToChatModel(dto);

        Assert.Equal(ClientChatRole.Assistant, model.Role);
    }

    [Fact]
    public void ToAskQuestionRequest_SetsRoleAndContent()
    {
        var createdAt = new DateTime(2026, 1, 2, 3, 4, 5, DateTimeKind.Utc);

        var request = ChatDtoMapper.ToAskQuestionRequest(
            content: "Ping",
            clientRole: ClientChatRole.User,
            createdAtUtc: createdAt);

        Assert.Equal(ApiChatRole._0, request.Role);
        Assert.Equal("Ping", request.Content);
        Assert.Equal(new DateTimeOffset(createdAt, TimeSpan.Zero), request.CreatedAt);
    }
}
