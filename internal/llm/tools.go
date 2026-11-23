package llm

import (
	"context"
	"fmt"
)

// ToolHandler is your real Go function behind a tool.
type ToolHandler func(ctx context.Context, args map[string]any) (map[string]any, error)

func Runtools (
	ctx context.Context,
	p Provider,
	history []Message,
	systemPrompt string,
	specs []ToolSpec,
	handlers map[string]ToolHandler,
	maxRounds int,
) (string, Usage, error) {
		msgs := append([]Message(nil), history...) // copy: the caller's slice stays untouched
	var total Usage

	for round := 0; round < maxRounds; round++ {
		reply, u, err := p.ChatTools(ctx, msgs, systemPrompt, specs)
		total.InputTokens += u.InputTokens
		total.OutputTokens += u.OutputTokens
		if err != nil {
			return "", total, err
		}

		if len(reply.ToolCalls) == 0 {
			return reply.Text, total, nil // the model answered; done
		}

		// keep the model's own turn exactly as received (signatures included)
		msgs = append(msgs, Message{Role: RoleAssistant, Content: reply.Text, ToolCalls: reply.ToolCalls})

		for _, call := range reply.ToolCalls {
			msgs = append(msgs, Message{
				Role:       RoleTool,
				ToolResult: &ToolResult{Name: call.Name, Result: runOne(ctx, handlers, call)},
			})
		}
	}

	return "", total, fmt.Errorf("tool loop did not finish after %d rounds", maxRounds)
}

func runOne(ctx context.Context, handlers map[string]ToolHandler, call ToolCall) map[string]any {
	h, ok := handlers[call.Name]
	if !ok {
		return map[string]any{"error": "unknown tool: " + call.Name}
	}
	out, err := h(ctx, call.Args)
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	if out == nil {
		out = map[string]any{}
	}
	return out
}