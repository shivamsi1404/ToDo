package llm

import "context"

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role    Role
	Content string
}

type Usage struct {
	InputTokens  int
	OutputTokens int
}

type Provider interface {
	// Chat sends a chat request to the LLM with the provided history and system prompt, returning the final response, usage statistics, and any error encountered.
	Chat(ctx context.Context, history []Message, systemPrompt string) (string, Usage, error)
	// ChatStream streams the response from the LLM, calling onChunk for each chunk of text received. It returns the final response, usage statistics, and any error encountered.
	ChatStream(ctx context.Context, history []Message, systemPrompt string, onChunk func(string)) (string, Usage, error)
}
