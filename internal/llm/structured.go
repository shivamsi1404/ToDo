package llm

import (
	"context"
	"fmt"
	"encoding/json"
)

func AskJSON[T any](ctx context.Context, p Provider, history []Message, systemPrompt string, schema map[string]any) (T, Usage, error) {
	
	// Initialize a variable to hold the total usage statistics
	var total Usage
	// Initialize a variable to hold the last error encountered during the attempts
	var lastErr error

	// Attempt to send the request up to 2 times
	for attempt := 0; attempt < 2; attempt++ {
		// Call the ChatJSON method of the provider with the provided context, history, system prompt, and schema
		raw, usage, err := p.ChatJSON(ctx, history, systemPrompt, schema)

		// Update the total usage statistics with the usage from the current attempt
		total.InputTokens += usage.InputTokens
		total.OutputTokens += usage.OutputTokens

		// If there was no error, attempt to unmarshal the raw response into the expected type T
		if err != nil {
			var zero T
			return zero, total, err // return early if there's an error from ChatJSON
		}

		// Unmarshal the raw response into the expected type T				
		var out T
		if err := json.Unmarshal([]byte(raw), &out); err != nil {
			lastErr = fmt.Errorf("failed to unmarshal response: %w", err)
			continue // retry on unmarshal error
		}
		return out, total, nil // return the successfully unmarshaled response and total usage
	}

	// If all attempts failed, return the last error encountered
	var zero T
	return zero, total, fmt.Errorf("gave up after 2 attempts: %w", lastErr) // return the last error encountered after all attempts
}

