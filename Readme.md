# ToDo

A command-line AI assistant written in Go, powered by the Google Gemini API. ToDo streams chat replies in your terminal, extracts structured tasks from plain text, and can call local tools (like reading your own notes) to answer questions.

## Features

- **Streaming chat**: replies print token by token, with conversation history kept across turns.
- **Structured task extraction** (`/json`): turns free text into a typed `Task` (title, priority, due date, tags) using Gemini's JSON response schema.
- **Tool calling** (`/tools`): the model can call local Go functions, with up to 5 tool rounds per question:
  - `get_current_time`: current local date, time and timezone
  - `list_notes`: lists your saved notes
  - `read_note`: reads one note from the `notes/` folder (path-traversal safe, capped at 20,000 bytes)
- **Resilient API client**: automatic retries with exponential backoff (1s, 2s, 4s) on `429` and `5xx` errors.
- **Safe cancellation**: press `Ctrl+C` to cancel a request; each request also times out after 2 minutes.
- **Token usage** shown after every reply.
- **Provider interface**: the LLM layer is abstracted behind `llm.Provider`, so other providers can be added later.

## Tech Stack

- Go
- Google Gemini API (REST, Server-Sent Events for streaming)
- [`godotenv`](https://github.com/joho/godotenv) for loading environment variables

## Project Structure

```
ToDo/
├── cmd/chat/          # CLI entry point (main package)
├── internal/
│   ├── chat/          # chat logic
│   └── llm/
│       ├── llm.go         # Provider interface and shared types
│       ├── gemini.go      # Gemini client (chat, stream, JSON, tools, retry)
│       ├── structured.go  # Generic AskJSON[T] helper
│       ├── tools.go       # Tool-calling loop (Runtools)
│       └── *_test.go      # Tests
├── notes/             # Markdown notes the assistant can read
├── .env               # Your API key (not committed)
├── go.mod
└── go.sum
```

## Getting Started

### Prerequisites

- Go (a recent version)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/shivamsi1404/ToDo.git
cd ToDo
go mod download
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

The model name is set in `main.go` (`llm.NewGemini(apiKey, "<model>")`), so change it there to use a different Gemini model.

### Run

```bash
go run ./cmd/chat
```

### Test

```bash
go test ./...
```

## Usage

| Input | What it does |
|-------|--------------|
| `any text` | Normal streaming chat |
| `/json <text>` | Extracts a structured task from your text |
| `/tools <question>` | Lets the assistant use its tools (time, notes) to answer |
| `/quit` | Exits the app |

### Examples

```
you> Explain goroutines in two lines
ToDo> ...

you> /json finish the OS lab report by Friday, high priority
task: {Title:Finish the OS lab report Priority:high Due:Friday Tags:[os lab]}

you> /tools what notes do I have?
[tool] list_notes map[]
ai> You have one note: me
```

### Using notes

Add Markdown files to the `notes/` folder (e.g. `notes/os-lab.md`). Then ask with `/tools`:

```
you> /tools summarise my os-lab note
```

## Roadmap

- [ ] **Persistent task storage**: save tasks in SQLite so they survive restarts
- [ ] **Task tools**: `add_task`, `list_tasks`, `complete_task` and `delete_task`, so the assistant manages your to-do list through tool calling
- [ ] **Saved conversations**: store chat history on disk and resume past sessions
- [ ] **More providers**: add OpenAI and Anthropic clients behind the existing `llm.Provider` interface
- [ ] **REST API**: expose chat and task endpoints over HTTP (Go `net/http`) alongside the CLI
- [ ] **CI and Docker**: GitHub Actions to run `go test ./...` and a Dockerfile for easy setup


## Author

**Shivam Singh**: [GitHub](https://github.com/shivamsi1404)