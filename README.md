1. Overview

Smart PDF reader is RAG application for reading large PDF files or batch of PDF files. It works as chat, where user does not need to read PDF files himself, but just ask applicaton
question from PDF files and application answer with the information from PDF. In small modifications, it could be also used with another media like wiki, confluence etc. 

Application is designed so that it runs on user's private server. LLM and embedding models are downloaded into local server. This solution provides with higher security
however there is no need to send sensitive data into third party infrastructure and everything is kept locally. Main disadvantage of this solution is higher requirement on 
the hardware performance. It would be better to run the application on GPU due to quite high latency when running on CPU (2-5 min per request).

2.Architecture
Application has three main layers: client, REST API, core RAG application. RAG application itself can be used with any other API. Communication with the RAG is secured using
fast API layer. REST API and client are implemented in .net. It has good tool for implementing REST API and user interface. The main purpose of using integration .net - python 
was demnostraion of integration pythonic AI solutions with .net framwerok solutions. 

Dataflow: 
user question - desktop client - REST API - RAG fast API - RAG 
									|						|
								SQL server 				 QDRANT	vector db

a) Desktop client - wrappping around rest api client generated from REST API (explained later). Client is chat window, where user write a question from the source document and 
send it into REST API. There is also possibility to delete whole conversation. The whole conversation history is downloaded on every turn on. There is configured limit of max
message count in conversation history.

b) REST API is built on standard REST API layer architecture : data+reposritory - service - web api (controllers etc.)
The main purpose of this module is to obtain message from client, process it and save into SQL DB. Then make request for fast API of RAG application and send it there. 
It also receiving response from RAG, saving it and forwarding into client. There is DELETE endpoint to reset the whole conversation history and GET endpoint to possibility
of loading the whole conversation history in the client after turning on. 

c) RAG application is the core part of the whole system. It has API layer provided by fast API, by which application communicate with the user. There are wto main functions of 
the RAG application: ingestion of PDFs and analysis of PDFs and responding questions about them. RAG applicationconsists of several layers:
ingestion part: loading document - chunking - embedding - saving into qdrant vector db
analyzing and responding part: embedding the query - qdrant similar search the most similar chunks - reranking - preparing prompt for llm - communication with the llm.
Bottleneck is llm part, however llm model is consuming a lot of HW performance. It is better to use it on GPU. CPU is causing quite high latency (2-5 min, sometimes more).

3. RAG pipeline
Ingestion:
a) Loading pdf files from configured directory and split them into chunks. Chunking has the main strategy to recognize chapters and subchapters and create tree system. Then 
find paragraphs inside of every chapter. If the each paragraph length is inside of configured limit interval, it is used as a chunk. If it is bigger, it is split into more chunks
and if it is shorter, it is merged with other chunk to satisfy the limit. There are few rules for splitting and merging the chunks. The main purpose of this workflow is the 
best possible semantic separation if the chunks.
In this phase, also payload of every chunk is created containing the most important metadata for the further processing. 
b)Embedding the chunks
c) Saving chunks into qdrant vector db with chunk id, vector from embedding and payload.

Responding questions pipeline:
a)question handling - in request from fast api there comming question with history of two recent messages (if provided). If the question contains reference on previous questions,
the history is used and appended into question. Of there is no reference on previous conversation, history is ignored.

b)Updatetd question is embedded using embedding model (currently configured multilingual-e5-small) 	

c)Getting semantically the most similar chunks from qdrant db using cosine search. Count of the selected most similar chunks is configured.

d)Getting just top two, the most similar chunks using reranking model (currently configured cross-encoder/ms-marco-MiniLM-L-6-v2).

e)Creating llm prompt using loaded chunks and history if needed. The current version of the prompt is in local configuration.

f)Calling llm model (currently configured phi3:mini, but there is also possibility of mistral mistral:latest).

4.RAG Architecture
RAG architecture is created from three layers, where each layer has it own purpose separated from othes
Infralayer (commuinicating and calling third party tools and db):
- db_client
- embedding_client
- llm_client
- reranking_client 		
Core (preparing data from upper layer and calling infralayer tools):
- pdf_parsing
- chunking
- embedding
- reranking
- llm_chatter
Application layer(communicate with the API and orchestrate lower layers calling Core layer):
- ingestion
- question_pipeline_orchestration
- rag_service (exposes RAG into API)
- main (console application for using RAG directly from the console and testing). 

Fallback: In the application, there is 2 step fallbacka strategy:
1. step: after reranking, there are selected the best chunks, which are semantically the closest to the question. But if even the chunk with the highest score does not reach the
configured limit of minimal score,processing is stopped and the response with the information that there was no information about that question available in the document is sent.
2. step: if the processing crosses successfully 1. step,  but selected chunks are still not completely relevant for the question, there is also llm responding for the prompt
instructed to announce in response, that there was no information in the document exactly relevant for the question. But none of fallback is 100% reliable, so sometimes 
there can appear some irrelevant answers for the questions which have nothing in common with the context of pdf document.

5. Issues
However the application was developed and used on quite poor HW (just weak CPU), there was big limitation for using strongers and more precise models. So fo embedding and reranking 
I have choosen the smallest models which could do the job on my infrastructure. Also as llm I tried mistral 7B model and also phi3:mini. However mistral had latency over 10-15 minutes
I needed to switch to smaller phi3:mini even despite of more precise reponses from mistral. The main issue connected to this is, that the accuracy of responses is sometimes lower.
Applications is trying to keep adding citations into every answer, but sometimes model "forgets" it. Also question targeting on information not present in documents sometimes 
cause false answers. FInally question refering to previous conversation does not work well due to this models. 

Key improvements: 
I have improved chunking process by improoving way how the document is splitted and how the paragraphs are merged together to keep good semantic separation of each paragraph. 
I have been experimenting with rewriting too, but the results were not good enough and it has risen the latency due to calling llm model one more time. I solved this issue 
by recognizing referencing words such as pronouns in the question and adding previous question and relevant sentences from the  previous answer into the current question
for embedding. In this case, the history is also appende into llm prompt.

6. Examples:
Ideal "information not available case":
User:Who is Rumcais?
Assistant:The information is not available in the provided document.

Response for question about document referring also to previous conversation:
User:What is the capital of Slovakia?
Assistant:Bratislava (Source: SVK-Welcome-to-Slovakia-booklet-2016; Chapter: The Slovak Republic - Population, Page: 5-6)
User:How many people live there?
Assistant:According to the SVK-Welcome-to-Slovakia-booklet-2016 (Source: SVK-Welcome-to-Slovakia-booklet-2016; Chapter: The Slovak Republic - Population,
Page: 5-6), Bratislava, the capital of Slovakia, has a population of approximately 500,000 inhabitants.

7.Tests:
For each module - RAG application, rest API and client, there are directories or projects specially prepared for tests. There are both:  unit and integration tests for each
layer. Integration means, that code is is tested in conditions of real application and check if everything before works with the current layer. 
In test mode, db connection is secured automatically using db_bootstrap.

There are also manual tests for each module. Application/App/main.py provides manual testing and possibility of debugging core RAG application. Rest API can be test through
swagger. 

8. Test metrics:
Reranking is part of RAG pipeline which assigns score to the chunks found in similar search, which would have the most similar meaning to the question.The higher score is,
the more similar meaning to the question the chunk has. 

I wanted to set threshold for excluding chunks with too low score from the collection of chunks for llm prompt. For that purpose I asked RAG 50 questions and rerank the chunks.
Results say that there is not exact score where the chunks would be strictly close to the question or would not. It is more the interval than point. Below, there are results from the test:
YES n= 89 min -9.9022 median 4.9849 max 9.5075
NO  n= 61 min -10.6536 median -1.3932 max 7.0882
This says, that min score of true chunk was -9,9, median was 4,99 and max score was 9,5. On the other hand, the minimal score of dalse chunk was -10,65, median was -1,39
and max was 7,09. That means, that retrieval is not good enough to implement any threshold. It somehow works, but there is quite wide interval of uncerntainty. Reason could be 
weak models for embedding and reranking and also weak heuristics and also that this is limit of the current architecture. For better results there should be added layer into 
question pipeline - search by keywords after cosine similar search and merging the results.  

Below is table of values of threshold and counts of FP (false positive) and FN (false negative) chunks separated by that threshold from -4 to 5:
T=-4.0: FP=41 FN=5
T=-3.0: FP=36 FN=6
T=-2.0: FP=34 FN=8
T=-1.0: FP=30 FN=11
T=0.0: FP=22 FN=13
T=1.0: FP=18 FN=18
T=2.0: FP=15 FN=23
T=3.0: FP=13 FN=27
T=4.0: FP=10 FN=36
T=5.0: FP=3 FN=45
The best value is with the most balanced counts of FP and FN, whoch is around T = 0.

Other metric obseved in the application is latency. The main bottleneck is llm model. While using mistral model it was around 5-10 min, however while using phi3:mini 
it was 2-5 min. Although phi3:mini is less precise and has more inaccuracies such as ingoring appending citations or incomplete citations sometimes or not following
prompt rules. It is better for interactive presentation of RAG system as the portfolio project. For better performance change to GPU HW would improve latency dramatically.

9.Set up and run

### Prerequisites (summary)

- **Python**: install deps from `requirements.txt` in a venv.
- **Docker**: used by `AI_module/dev_tools/start_app_db.bat` to run Qdrant.
- **Qdrant**: vector DB at `http://localhost:6333`.
- **Ollama**: local LLM server. Pull the configured model (see `AI_module/config.py` → `llm_model`, default `phi3:mini`).
- **SQL Server**: used by `SmartPdfReaderApi` (see `SmartPdfReaderApi/SmartPdfReaderApi/appsettings.json`).
- **.NET SDKs**:
  - `SmartPdfReaderApi` targets **.NET 8**
  - `DesktopClient` targets **.NET 9 (Windows/WPF)**

### One-time preparation (recommended)

- **Embedding model offline snapshot** (so embedding tests can run offline):
  - Run once while online: `AI_module\dev_tools\download_embedding_model.bat`
  - This downloads the Hugging Face model snapshot into `AI_module/data/models/...` and the app/tests will use the local folder when present.

- **SQL DB migrations**
  - The EF Core migrations live in `SmartPdfReaderApi/Data/Migrations`.
  - Apply them using the provided script:
    - `SmartPdfReaderApi\Data\apply_migrations.bat`
  - This uses `dotnet ef` under the hood and reads the connection string from `SmartPdfReaderApi/SmartPdfReaderApi/appsettings.json`.

### PDF ingestion (load PDFs into Qdrant)

1. Put PDFs into `AI_module/data/pdfs` (config: `PDF_INPUT_DIR` in `AI_module/config.py`).
2. Start Qdrant.
3. Run ingestion:
   - `AI_module\dev_tools\run_ingestion.bat`

### Run variants

#### Variant 1 — Start Qdrant and RAG separately (plus run .NET API manually)

1. Start Qdrant: `AI_module\dev_tools\start_app_db.bat`
2. Start RAG FastAPI: `AI_api\run_rag.bat` (docs: `http://localhost:8000/docs`)
3. Start .NET REST API: `dotnet run --project "SmartPdfReaderApi/SmartPdfReaderApi/SmartPdfReaderApi.csproj"`
4. Run ingestion (optional): `AI_module\dev_tools\run_ingestion.bat`
5. Run DesktopClient (WPF, Windows).

Stop:
- RAG: `AI_api\stop_rag.bat`
- Qdrant: `AI_module\dev_tools\stop_app_db.bat`
- .NET API: stop the `dotnet run` process

#### Variant 2 — Start core stack (Windows convenience script)

From repo root:
- `start_core_app.bat` (starts Qdrant, RAG FastAPI, and SmartPdfReaderApi)

Stop:
- `stop_core_app.bat`

#### Variant 3 — Manual / cross-platform

1. Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
2. RAG: `uvicorn AI_api.main:app --host 0.0.0.0 --port 8000`
3. .NET API: `dotnet run --project SmartPdfReaderApi/SmartPdfReaderApi/SmartPdfReaderApi.csproj`

### Ports (defaults)

- Qdrant: `http://localhost:6333`
- RAG FastAPI: `http://localhost:8000`
- SmartPdfReaderApi: `http://localhost:5000`

10. Regenerating desktop clinet in case of changes in REST API:
### Regenerating DesktopClient Swagger Client (NSwag)

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

10.Future improvements:
Upgrading HW to GPU would definitely help to improve performance of application. Better HW would allow use better llm and also embeding models, so the latency would be shorter
and system would be more precise. It maybe allow add rewrite function using llm to get better results in case of question referencing to the previous messages. Current chunking
 heuristics also could be improved, which finally could lead to more clear and shorter prompts and therefore shorter models processing and more precise results and less halucinations. 
