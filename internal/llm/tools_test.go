package llm

import (
	"context"
	"testing"
)

type scriptedProvider struct {
	replies []Reply
	seen    [][]Message
}

func (s *scriptedProvider) Chat(context.Context, []Message, string) (string, Usage, error) {
	return "", Usage{}, nil
}
func (s *scriptedProvider) ChatStream(context.Context, []Message, string, func(string)) (string, Usage, error) {
	return "", Usage{}, nil
}
func (s *scriptedProvider) ChatJSON(context.Context, []Message, string, map[string]any) (string, Usage, error) {
	return "", Usage{}, nil
}
func (s *scriptedProvider) ChatTools(_ context.Context, history []Message, _ string, _ []ToolSpec) (Reply, Usage, error) {
	s.seen = append(s.seen, append([]Message(nil), history...))
	r := s.replies[0]
	s.replies = s.replies[1:]
	return r, Usage{InputTokens: 1, OutputTokens: 1}, nil
}

func TestRunToolsHappyPath(t *testing.T) {
	p := &scriptedProvider{replies: []Reply{
		{ToolCalls: []ToolCall{{Name: "echo", Args: map[string]any{"x": "hi"}, Opaque: "SIG"}}},
		{Text: "done"},
	}}
	handlers := map[string]ToolHandler{
		"echo": func(_ context.Context, args map[string]any) (map[string]any, error) { return args, nil },
	}

	text, usage, err := Runtools(context.Background(), p,
		[]Message{{Role: RoleUser, Content: "go"}}, "", nil, handlers, 5)
	if err != nil || text != "done" {
		t.Fatalf("got %q, %v", text, err)
	}
	if usage.InputTokens != 2 || usage.OutputTokens != 2 {
		t.Fatalf("usage not summed: %+v", usage)
	}
	if len(p.seen) != 2 {
		t.Fatalf("expected 2 model calls, got %d", len(p.seen))
	}

	second := p.seen[1] // what the model saw on the second call
	if len(second) != 3 {
		t.Fatalf("expected user, assistant, tool; got %d messages", len(second))
	}
	if second[1].ToolCalls[0].Opaque != "SIG" {
		t.Fatalf("signature was lost: %+v", second[1])
	}
	if second[2].Role != RoleTool || second[2].ToolResult.Result["x"] != "hi" {
		t.Fatalf("tool result missing: %+v", second[2])
	}
}

func TestRunToolsStopsAtCap(t *testing.T) {
	call := Reply{ToolCalls: []ToolCall{{Name: "echo"}}}
	p := &scriptedProvider{replies: []Reply{call, call}}
	handlers := map[string]ToolHandler{
		"echo": func(context.Context, map[string]any) (map[string]any, error) { return nil, nil },
	}

	if _, _, err := Runtools(context.Background(), p, nil, "", nil, handlers, 2); err == nil {
		t.Fatal("expected an error after hitting the round cap")
	}
}