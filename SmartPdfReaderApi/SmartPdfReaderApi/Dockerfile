FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy csproj files first (better caching)
COPY SmartPdfReaderApi/SmartPdfReaderApi/SmartPdfReaderApi.csproj SmartPdfReaderApi/SmartPdfReaderApi/
COPY SmartPdfReaderApi/Data/Data.csproj SmartPdfReaderApi/Data/
COPY SmartPdfReaderApi/Service/Service.csproj SmartPdfReaderApi/Service/
RUN dotnet restore SmartPdfReaderApi/SmartPdfReaderApi/SmartPdfReaderApi.csproj

# Copy everything else and publish
COPY . .
RUN dotnet publish SmartPdfReaderApi/SmartPdfReaderApi/SmartPdfReaderApi.csproj -c Release -o /out /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /out .

ENV ASPNETCORE_URLS=http://0.0.0.0:5000
EXPOSE 5000

ENTRYPOINT ["dotnet", "SmartPdfReaderApi.dll"]
