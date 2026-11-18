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
    Chat(ctx context.Context, history []Message, systemPrompt string) (string, Usage, error)
}