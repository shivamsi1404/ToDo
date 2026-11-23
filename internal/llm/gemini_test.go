package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildRequestToolTurns(t *testing.T) {
	history := []Message{
		{Role: RoleUser, Content: "what time is it?"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{Name: "get_current_time", Opaque: "SIG"}}},
		{Role: RoleTool, ToolResult: &ToolResult{Name: "get_current_time", Result: map[string]any{"time": "14:05"}}},
	}

	b, err := json.Marshal(buildRequest(history, ""))
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)

	for _, want := range []string{`"functionCall"`, `"thoughtSignature":"SIG"`, `"functionResponse"`, `"role":"model"`, `"args":{}`} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %s in %s", want, s)
		}
	}
	if strings.Contains(s, `"text":""`) {
		t.Errorf("empty text part in %s", s)
	}
}

func TestParseReply(t *testing.T) {
	raw := []byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_current_time","args":{}},"thoughtSignature":"SIG"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`)

	reply, usage, err := parsereply(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(reply.ToolCalls) != 1 || reply.ToolCalls[0].Name != "get_current_time" || reply.ToolCalls[0].Opaque != "SIG" {
		t.Fatalf("bad tool call: %+v", reply)
	}
	if reply.Text != "" || usage.InputTokens != 5 {
		t.Fatalf("unexpected: %+v %+v", reply, usage)
	}

	reply, _, err = parsereply([]byte(`{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}`))
	if err != nil || reply.Text != "hi" || len(reply.ToolCalls) != 0 {
		t.Fatalf("text reply failed: %+v %v", reply, err)
	}
}