# Technical Report — Puls-Events RAG Assistant

## 1. Introduction and Business Context

Puls-Events is an innovative company in the cultural events sector. Its platform helps users discover cultural events and filter them by location and time period. The objective of this project is to create a first Proof of Concept for an intelligent assistant capable of recommending cultural events using real event data.

The project is based on a Retrieval-Augmented Generation architecture. Instead of allowing the language model to answer from general knowledge, the system first retrieves relevant events from a vector database and then generates an answer grounded in those retrieved events.

The selected data source is OpenAgenda, accessed through the public OpenDataSoft API. For this POC, the geographic scope was limited to Paris. This narrow scope makes the system easier to test, evaluate, and demonstrate.

The final system is not designed as a full production application. It is a functional and reproducible POC showing that OpenAgenda data can be collected, cleaned, indexed, searched semantically, and used by a chatbot to generate relevant cultural event recommendations.

---

## 2. Project Objective

The main objective is to prove that a RAG system can recommend cultural events using real, recent, and geographically filtered OpenAgenda data.

The POC must demonstrate that:

* cultural event data can be collected from OpenAgenda;
* events can be filtered by city and by recency;
* event descriptions can be cleaned and transformed into searchable text;
* text chunks can be embedded using a Mistral embedding model;
* FAISS can retrieve semantically relevant events;
* Mistral can generate natural language answers using only the retrieved context;
* the system can be tested and evaluated using automated tests and an annotated Q/A dataset;
* the system can be demonstrated through both a CLI chatbot and a Streamlit web interface.

The project scope is:

```text
Geographic scope: Paris
Data source: OpenAgenda public dataset via OpenDataSoft
Time scope: recent / active / upcoming events
Vector database: FAISS
Embedding model: mistral-embed
Chat model: mistral-small-latest
Framework: LangChain
Demo interface: CLI + Streamlit
```

---

## 3. Global System Architecture

The architecture follows a classical RAG pipeline:

```text
OpenAgenda API
      ↓
Data preprocessing
      ↓
Clean event DataFrame
      ↓
LangChain Documents
      ↓
Text chunks
      ↓
Mistral embeddings
      ↓
FAISS vector index
      ↓
Retriever
      ↓
Mistral chat model
      ↓
Final grounded answer
```

The system is divided into several independent modules:

```text
src/data/preprocess_openagenda.py
    Fetches, cleans, filters, and structures OpenAgenda events.

src/index/build_faiss.py
    Converts clean events into LangChain Documents, creates embeddings,
    builds the FAISS index, and saves it locally.

src/rag/rag_pipeline.py
    Loads the FAISS index and runs the RAG chatbot from the command line.

src/evaluation/evaluate_rag.py
    Evaluates the chatbot using an annotated French Q/A dataset.

app.py
    Provides a Streamlit web interface for the live demo.

tests/test_data_quality.py
    Contains automated tests for data quality and FAISS index validation.
```

This modular design makes the project reproducible and easy to explain. Each part of the system has a clear role.

---

## 4. Data Ingestion from OpenAgenda

The project uses the OpenAgenda public event dataset available through the OpenDataSoft Explore API.

The ingestion step is implemented in:

```text
src/data/preprocess_openagenda.py
```

The script sends API requests to retrieve events for a selected city. Pagination is handled using `limit` and `offset`, with a maximum of 100 records per request.

For this POC, the default city is Paris. The API query filters events using the city name and a recency condition.

A specific decision was made regarding event dates. Some events are recurring or spread across multiple dates. For example, an event may have a first start date older than one year, but still have recent or future occurrences. For this reason, the preprocessing logic uses `lastdate_begin` internally to decide whether an event is still relevant.

The final returned dataset does not expose this internal column. It only returns the clean project schema.

Final schema:

```text
uid
title
city
postal_code
venue
url
tags
start_utc
end_utc
text
```

---

## 5. Data Preprocessing and Quality Controls

The preprocessing step prepares the event data for indexing.

The main operations are:

* filtering by selected city;
* keeping recent, active, or upcoming events;
* parsing start and end dates as UTC timestamps;
* cleaning HTML descriptions using BeautifulSoup;
* normalizing empty or missing text fields;
* normalizing tags;
* removing events with missing IDs, titles, dates, or descriptions;
* removing duplicate events using the `uid` field;
* returning a stable schema.

The cleaned text field is especially important because it becomes the main content embedded into vectors.

The final clean DataFrame is not saved automatically by the preprocessing script. This is intentional. The preprocessing module is used directly by the FAISS build script. The FAISS build step is responsible for creating and saving the vector index.

Data quality was validated with automated tests using `pytest`.

The tests check:

* the clean dataset is not empty;
* the final schema is correct;
* internal filtering columns are not returned;
* all events belong to the selected city;
* important fields such as `uid`, `title`, and `text` are not empty;
* start and end dates are valid;
* events are recent, active, or upcoming;
* FAISS index files exist;
* the FAISS index can be loaded successfully.

Latest test result:

```text
9 passed
```

These tests show that the pipeline respects the main data quality constraints of the mission.

---

## 6. Vectorization and FAISS Indexing

The vectorization and indexing step is implemented in:

```text
src/index/build_faiss.py
```

This script calls the preprocessing module, converts each event into a LangChain `Document`, splits long documents into chunks, generates embeddings with Mistral, and stores the result in FAISS.

Each event document contains both text content and metadata.

The embedded text includes:

```text
Title
City
PostalCode
Venue
Start date
End date
YearStart
MonthStart
YearEnd
MonthEnd
Tags
URL
Description
```

The `YearStart`, `MonthStart`, `YearEnd`, and `MonthEnd` tokens were added to improve retrieval for date-related questions such as:

```text
events this month
events in May
events in 2026
```

The postal code is also included to improve local search inside Paris.

The text is split using `RecursiveCharacterTextSplitter` with the following default values:

```text
chunk_size = 800
chunk_overlap = 120
```

The embedding model used is:

```text
mistral-embed
```

The FAISS index is saved locally in:

```text
data/index/faiss/
```

The output contains two files:

```text
index.faiss
index.pkl
```

`index.faiss` contains the numerical vector index used for similarity search.

`index.pkl` contains the LangChain document store and metadata, allowing the system to map retrieved vectors back to event titles, dates, venues, URLs, and descriptions.

The index can be rebuilt on demand with:

```bash
python -m src.index.build_faiss --city Paris --max-records 1000 --index-out data/index/faiss
```

This satisfies the reproducibility requirement of the project.

---

## 7. RAG Pipeline with LangChain and Mistral

The RAG chatbot is implemented in:

```text
src/rag/rag_pipeline.py
```

The chatbot follows this flow:

```text
User question
      ↓
FAISS similarity search
      ↓
Retrieved event chunks
      ↓
Context formatting
      ↓
Prompt construction
      ↓
Mistral chat model
      ↓
Grounded answer
```

The chatbot loads the saved FAISS index using the same embedding model used at build time. This is important because the query embedding must be created in the same vector space as the indexed documents.

The chatbot uses a strict system prompt. The model is instructed to:

* answer only using the provided context;
* not invent events, dates, venues, prices, or links;
* say that it does not know when the context does not contain enough information;
* reply in the same language as the user’s question.

The selected chat model is:

```text
mistral-small-latest
```

The model temperature is set to:

```text
temperature = 0.1
```

A low temperature was chosen to make the answers more stable and factual.

The project also includes a Streamlit interface:

```text
app.py
```

The Streamlit app allows the user to:

* enter a question in the browser;
* retrieve relevant event chunks from FAISS;
* generate an answer with Mistral;
* view the retrieved sources used by the model.

This makes the POC easier to demonstrate during the oral presentation.

---

## 8. Evaluation Method and Results

The project includes an annotated French Q/A dataset:

```text
data/test/qa_test_set.csv
```

The dataset contains 15 French questions covering different user needs:

* children’s events;
* family activities;
* music events;
* exhibitions;
* free events;
* nature and gardens;
* creative workshops;
* cultural visits;
* science and educational activities;
* events with URLs;
* out-of-scope city queries.

The evaluation script is:

```text
src/evaluation/evaluate_rag.py
```

The script performs the following steps:

```text
1. Read the annotated Q/A dataset.
2. Load the FAISS index.
3. Ask each question to the RAG system.
4. Generate an answer with Mistral.
5. Compare the generated answer with expected keywords.
6. Save the results in evaluation_results.csv.
```

The evaluation is keyword-based. This is a simple and transparent method suitable for a POC. It does not replace human evaluation, but it provides a measurable first indication of answer quality.

Latest evaluation result:

```text
Passed: 14/15
Score: 93.33%
```

The only failed case was an out-of-scope geographic query:

```text
Y a-t-il des événements à Lyon dans cet index ?
```

The FAISS index was built for Paris, but the system still retrieved semantically close Paris events. This shows a limitation of the current POC: the retriever does not yet enforce strict geographic validation before retrieval.

This failure is useful because it identifies a clear improvement for a production version: add explicit city detection and validation before querying the vector index.

---

## 9. Live Demo

The project can be demonstrated in two ways.

### CLI Demo

The CLI chatbot can be launched with:

```bash
python -m src.rag.rag_pipeline --index data/index/faiss --k 8
```

Example questions:

```text
Recommande-moi une activité culturelle pour des enfants à Paris.
Trouve-moi un événement musical à Paris.
Y a-t-il des événements gratuits à Paris ?
Peux-tu me proposer une exposition à Paris ?
```

### Streamlit Demo

The web demo can be launched with:

```bash
streamlit run app.py
```

The interface opens in the browser at:

```text
http://localhost:8501
```

The Streamlit app displays:

* the user question input;
* the generated answer;
* the retrieved sources;
* metadata such as title, city, venue, dates, and URL.

This live demo proves that the system is functional and usable by non-technical users.

---

## 10. Technical Choices

### Why OpenAgenda?

OpenAgenda was selected because it is the required public data source for the mission. It provides real cultural event data and contains useful fields such as titles, descriptions, cities, venues, dates, tags, and URLs.

### Why Pandas?

Pandas was used for preprocessing because it provides a simple and efficient way to clean, filter, normalize, and validate tabular event data.

### Why LangChain?

LangChain was used to connect the different components of the RAG pipeline:

```text
documents
text splitter
embedding model
vector store
retriever
prompt template
chat model
```

It makes the system easier to structure and maintain.

### Why Mistral?

Mistral was used for both embeddings and answer generation. This keeps the project coherent and allows the system to use a modern LLM provider for both semantic vectorization and natural language generation.

### Why FAISS?

FAISS was selected because it is fast, local, and well suited for a POC. It allows semantic similarity search over event embeddings without requiring a managed vector database.

### Why Streamlit?

Streamlit was selected for the live demo because it is simple, fast to implement, and suitable for presenting machine learning or data applications in a browser.

---

## 11. Limitations of the POC

This project is functional, but it remains a Proof of Concept.

The main limitations are:

1. **Local FAISS index**

   The FAISS index is stored locally. This is sufficient for a POC, but a production system may require cloud storage, monitoring, backups, and possibly a managed vector database.

2. **Small evaluation dataset**

   The Q/A dataset contains 15 questions. This is enough for a first evaluation, but production evaluation would require a larger and more diverse dataset.

3. **No strict geographic validation before retrieval**

   The failed Lyon question shows that the system should detect when the user asks about a city outside the indexed scope.

4. **No deduplication of retrieved events**

   FAISS retrieves chunks, not necessarily unique events. Several retrieved chunks may correspond to the same event.

5. **No conversation memory**

   The chatbot does not keep conversation history. This is acceptable for the POC, but a production assistant may need memory for follow-up questions.

6. **Dependency on Mistral API availability**

   The system depends on the availability, latency, and cost of the Mistral API.

7. **OpenAgenda data quality**

   Event descriptions, dates, categories, and URLs depend on the quality of the source data. Missing or incomplete source fields can affect answer quality.

8. **Limited production monitoring**

   The POC does not yet include logging, user feedback, drift monitoring, cost tracking, or alerting.

---

## 12. Recommendations for Production

To move from POC to production, several improvements are recommended.

### 1. Add city validation

Before retrieval, the system should detect the requested city and verify whether an index exists for that city.

Example:

```text
User asks about Lyon
System checks available indexes
If only Paris index exists:
    answer that Lyon is not available
```

### 2. Use one index per city or region

Instead of one generic index, the system could maintain separate indexes:

```text
data/index/faiss/paris
data/index/faiss/lyon
data/index/faiss/strasbourg
```

This would improve geographic precision.

### 3. Deduplicate retrieved events

The retriever should retrieve more chunks than needed and then deduplicate results by `uid`.

Example:

```text
retrieve 25 chunks
deduplicate by event uid
show 5 to 10 unique events
```

### 4. Schedule automatic index updates

OpenAgenda data changes over time. A production system should rebuild or update the index regularly, for example daily or weekly.

### 5. Add stronger evaluation

Future evaluation should include:

```text
retrieval precision
retrieval recall
answer correctness
hallucination rate
user satisfaction
latency
cost per query
```

### 6. Add monitoring and logging

Production monitoring should track:

```text
failed answers
unanswered questions
retrieval quality
API errors
latency
token usage
embedding cost
chat model cost
data freshness
```

### 7. Improve the user interface

The Streamlit interface could be improved with:

```text
city selector
date filters
event category filters
number of events requested
source cards
feedback buttons
```

### 8. Consider a managed vector database

FAISS is excellent for a local POC. For larger-scale deployment, a managed vector database or cloud-based vector search system may be more suitable.

---

## 13. Conclusion

This project successfully demonstrates a functional RAG Proof of Concept for cultural event recommendations.

The system can:

* collect OpenAgenda event data;
* filter events by city and recency;
* clean and structure event descriptions;
* create LangChain Documents;
* split text into chunks;
* generate Mistral embeddings;
* build and save a FAISS vector index;
* retrieve relevant events using semantic search;
* generate grounded answers with Mistral;
* run as a CLI chatbot;
* run as a Streamlit web application;
* validate data quality with automated tests;
* evaluate answers using a French annotated Q/A dataset.

The POC achieved:

```text
Unit tests: 9 passed
Evaluation score: 14/15
Success rate: 93.33%
```

The main limitation identified during evaluation is geographic scope handling. The system works well for Paris, but it should explicitly detect and reject or redirect questions about cities not included in the current index.

Overall, the project demonstrates that a RAG architecture using OpenAgenda, LangChain, Mistral, and FAISS can provide relevant cultural event recommendations from real-world data. It is therefore a solid basis for a future production version of the Puls-Events assistant.

---

