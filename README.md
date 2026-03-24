# smart_pdf_reader
Application reads large PDF files and is able to answer questions about topics of this PDF in chat.

## Regenerating DesktopClient Swagger Client (NSwag)

DesktopClient includes a generated HTTP client and DTOs for `SmartPdfReaderApi`. They are generated from Swagger at `http://localhost:5000/swagger/v1/swagger.json`.

### Prerequisites

1. Start `SmartPdfReaderApi` in Development (so Swagger is enabled):
   - `dotnet run --project "SmartPdfReaderApi/SmartPdfReaderApi/SmartPdfReaderApi.csproj" --launch-profile "SmartPdfReaderApi"`
2. Verify Swagger is reachable:
   - `http://localhost:5000/swagger/v1/swagger.json`

### Install NSwag (once)

```powershell
dotnet tool install --global nswag.consolecore
```

### Regenerate client + DTOs

```powershell
nswag openapi2csclient `
  /input:"http://localhost:5000/swagger/v1/swagger.json" `
  /output:"DesktopClient/DesktopClient/Client/SmartPdfReaderApiClient.cs" `
  /namespace:"DesktopClient.Client" `
  /className:"SmartPdfReaderApiClient" `
  /GenerateClientInterfaces:true `
  /InjectHttpClient:true `
  /DisposeHttpClient:false `
  /GenerateContractsOutput:true `
  /ContractsNamespace:"DesktopClient.ModelsDTO" `
  /ContractsOutput:"DesktopClient/DesktopClient/ModelsDTO/SmartPdfReaderApiDtos.cs" `
  /GenerateDtoTypes:true `
  /JsonLibrary:SystemTextJson
```

Notes:
1. `DesktopClient/DesktopClient/ModelsDTO/ChatDtoMapper.cs` is handwritten and should not be regenerated/overwritten.
2. Generated DTOs are in `DesktopClient.ModelsDTO`, and the client is in `DesktopClient.Client`.
