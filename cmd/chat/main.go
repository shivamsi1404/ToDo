package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/joho/godotenv"

	"github.com/shivamsi1404/ToDo/internal/llm"
)

type Task struct {
	Title    string   `json:"title"`
	Priority string   `json:"priority"`
	Due      string   `json:"due"`
	Tags     []string `json:"tags"`
}

var taskSchema = map[string]any{
	"type": "OBJECT",
	"properties": map[string]any{
		"title":    map[string]any{"type": "STRING"},
		"priority": map[string]any{"type": "STRING", "enum": []string{"low", "medium", "high"}},
		"due":      map[string]any{"type": "STRING"},
		"tags":     map[string]any{"type": "ARRAY", "items": map[string]any{"type": "STRING"}},
	},
	"required": []string{"title", "priority"},
}

// real tool

// the notes directory is where the user's notes are stored. It should be a folder named "notes" in the current working directory.
const notesDir = "notes"

// getCurrentTime returns the current local date and time in a structured format.
func getCurrentTime(ctx context.Context, args map[string]any) (map[string]any, error) {
	now := time.Now()
	return map[string]any{
		"date":     now.Format("2006-01-02"),
		"time":     now.Format("15:04:05"),
		"timezone": now.Location().String(),
	}, nil
}

// listNotes lists the names of the user's saved notes in the notes directory. It returns a map containing the note names without the ".md" extension.
func listNotes(ctx context.Context, args map[string]any) (map[string]any, error) {
	entries, err := os.ReadDir(notesDir)
	if err != nil {
		return nil, fmt.Errorf("notes folder not found")
	}
	names := []string{}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".md") {
			names = append(names, strings.TrimSuffix(e.Name(), ".md"))
		}
	}
	return map[string]any{"notes": names}, nil
}

// readNote reads the content of a note by name and returns it in a structured format.
func readNote(ctx context.Context, args map[string]any) (map[string]any, error) {
	name, _ := args["name"].(string)
	name = strings.TrimSuffix(strings.TrimSpace(name), ".md")

	// only a plain file name is allowed: no folders, no "..", no absolute paths
	if name == "" || name != filepath.Base(name) || strings.Contains(name, "..") {
		return nil, fmt.Errorf("invalid note name")
	}

	data, err := os.ReadFile(filepath.Join(notesDir, name+".md"))
	if err != nil {
		return nil, fmt.Errorf("could not read note %q", name)
	}

	const maxBytes = 20000 // keep one note from flooding the context window
	text := string(data)
	if len(text) > maxBytes {
		text = text[:maxBytes]
	}
	return map[string]any{"name": name, "content": text}, nil
}

// toolSpecs and toolHandlers define the available tools and their corresponding Go functions. These are used by the LLM to request information or perform actions.
var toolSpecs = []llm.ToolSpec{
	{Name: "get_current_time", Description: "Get the current local date and time."},
	{Name: "list_notes", Description: "List the names of the user's saved notes."},
	{
		Name:        "read_note",
		Description: "Read one saved note by name, without the file extension. Use list_notes first if unsure of the name.",
		Parameters: map[string]any{
			"type": "OBJECT",
			"properties": map[string]any{
				"name": map[string]any{"type": "STRING", "description": "Note name, e.g. os-lab"},
			},
			"required": []string{"name"},
		},
	},
}

func logged(name string, h llm.ToolHandler) llm.ToolHandler {
	return func(ctx context.Context, args map[string]any) (map[string]any, error) {
		fmt.Printf("[tool] %s %v\n", name, args)
		return h(ctx, args)
	}
}

var toolHandlers = map[string]llm.ToolHandler{
	"get_current_time": logged("get_current_time", getCurrentTime),
	"list_notes":       logged("list_notes", listNotes),
	"read_note":        logged("read_note", readNote),
}

func main() {
	_ = godotenv.Load() // fine if .env is missing and the key is already in the environment

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("GEMINI_API_KEY is not set")
		os.Exit(1)
	}

	var provider llm.Provider = llm.NewGemini(apiKey, "gemini-3.5-flash-lite")
	systemPrompt := "You are ToDo. Act like my personal assistant reply in a helpful and informative way but limited in length. Treat me as your boss and stay as my main guide. You are a helpful assistant that provides information and answers questions. You should be concise and to the point, but also provide enough detail to be helpful. You should also be polite and respectful. You should provide personal opinions or advice but with proper facts."

	history := []llm.Message{}
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("Chat started. Type /quit to exit.")
	for {
		fmt.Print("\nyou> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "/quit" {
			break
		}

		// ---- new block starts here ----
		if strings.HasPrefix(line, "/json ") {
			text := strings.TrimPrefix(line, "/json ")

			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			task, usage, err := llm.AskJSON[Task](
				ctx,
				provider,
				[]llm.Message{{Role: llm.RoleUser, Content: text}},
				"Extract one task from the user's text. Leave due empty if no date is given.",
				taskSchema,
			)
			cancel()

			if err != nil {
				fmt.Println("error:", err)
				continue
			}

			fmt.Printf("task: %+v\n[tokens: in=%d out=%d]\n", task, usage.InputTokens, usage.OutputTokens)
			continue
		}
		// ---- new block ends here ----

		// ---- tool block starts here ----
		if strings.HasPrefix(line, "/tools ") {
			question := strings.TrimPrefix(line, "/tools ")

			// tool turns live in a copy; the main history only gets the question and the final answer
			msgs := append(append([]llm.Message(nil), history...), llm.Message{Role: llm.RoleUser, Content: question})

			sigCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
			ctx, cancel := context.WithTimeout(sigCtx, 2*time.Minute)
			text, usage, err := llm.Runtools(ctx, provider, msgs, systemPrompt, toolSpecs, toolHandlers, 5)
			ctxErr := ctx.Err()
			cancel()
			stop()

			if err != nil {
				switch {
				case errors.Is(ctxErr, context.Canceled):
					fmt.Println("[cancelled]")
				case errors.Is(ctxErr, context.DeadlineExceeded):
					fmt.Println("[timed out]")
				default:
					fmt.Println("error:", err)
				}
				continue
			}

			history = append(history,
				llm.Message{Role: llm.RoleUser, Content: question},
				llm.Message{Role: llm.RoleAssistant, Content: text},
			)
			fmt.Printf("ai> %s\n[tokens: in=%d out=%d]\n", text, usage.InputTokens, usage.OutputTokens)
			continue
		}

		// ---- tool block ends here ----

		history = append(history, llm.Message{Role: llm.RoleUser, Content: line})

		sigctx, stop := signal.NotifyContext(context.Background(), os.Interrupt) //
		ctx, cancel := context.WithTimeout(sigctx, 2*time.Minute)                // set a timeout for the chat request

		fmt.Print("ToDo> ")
		reply, usage, err := provider.ChatStream(ctx, history, systemPrompt, func(s string) {
			fmt.Print(s)
		})

		ctxerr := ctx.Err() // check if the context was canceled or timed out
		cancel()            // after the call returns, not inside the callback
		stop()              // stop listening for signals
		fmt.Println()

		if err != nil {
			history = history[:len(history)-1] // remove the last user message on error
			switch {
			case errors.Is(ctxerr, context.Canceled): // check if the context was canceled
				fmt.Println("Chat canceled by user.")
			case errors.Is(ctxerr, context.DeadlineExceeded): // check if the context timed out
				fmt.Println("Chat timed out.")
			default:
				fmt.Printf("Error: %v\n", err)
			}
			continue
		}

		history = append(history, llm.Message{Role: llm.RoleAssistant, Content: reply})
		fmt.Printf("[tokens: in=%d out=%d]\n", usage.InputTokens, usage.OutputTokens)
	}
	fmt.Println("Chat ended.")

	if err := scanner.Err(); err != nil {
		fmt.Println("error reading input:", err)
	}
}
