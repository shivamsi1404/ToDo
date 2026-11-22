package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type GeminiRequest struct {
	SystemInstruction *Content  `json:"systemInstruction,omitempty"`
	Contents          []Content `json:"contents"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
}

type GenerationConfig struct {
	ResponseMimeType string         `json:"responseMimeType,omitempty"`
	ResponseSchema   map[string]any `json:"responseSchema,omitempty"`
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

type Part struct {
	Text string `json:"text"`
}

type GeminiResponse struct {
	Candidates    []Candidate   `json:"candidates"`
	UsageMetadata UsageMetadata `json:"usageMetadata"`
}

type Candidate struct {
	Content      Content `json:"content"`
	FinishReason string  `json:"finishReason"`
}

type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
}

var _ Provider = (*Gemini)(nil) // Ensure Gemini implements the Provider interface

// Chat method for the Gemini struct
func (g *Gemini) generate(ctx context.Context, body GeminiRequest) (string, Usage, error) {

	// get the endpoint for the Gemini API
	endpoint := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent",
		g.model,
	)

	// marshal the request body into JSON format
	jsondata, err := json.Marshal(body)
	if err != nil {
		return "", Usage{}, err
	}

	// send the request to the Gemini API with retry logic
	response, err := g.doWithRetry(ctx, endpoint, jsondata)
	if err != nil {
	return "", Usage{}, err
	}

	// ensure the response body is closed after reading to avoid resource leaks
	defer response.Body.Close()

	// read the response body raw and print it for debugging purposes
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return "", Usage{}, err
	}
	// fmt.Println("Raw response from Gemini API:", string(raw)) // Uncomment this line for debugging purposes

	// unmarshal the response body into a GeminiResponse struct
	var result GeminiResponse

	// unmarshal the raw response into the result struct and check for errors
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", Usage{}, err
	}

	// check if the result has any candidates, if not return an error
	if len(result.Candidates) == 0 {
		return "", Usage{}, fmt.Errorf("gemini returned no candidates")
	}

	// get the first candidate from the result and check if it has any content parts
	cand := result.Candidates[0]
	if len(cand.Content.Parts) == 0 {
		return "", Usage{}, fmt.Errorf("gemini returned no content (finish reason: %s)", cand.FinishReason)
	}

	// concatenate all the text parts into a single string
	var text strings.Builder

	// iterate over the content parts and write the text to the string builder
	for _, p := range cand.Content.Parts {
		text.WriteString(p.Text)
	}

	// create a Usage struct with the input and output token counts from the result
	usage := Usage{
		InputTokens:  result.UsageMetadata.PromptTokenCount,
		OutputTokens: result.UsageMetadata.CandidatesTokenCount,
	}

	// return the concatenated text, usage, and nil error
	return text.String(), usage, nil
}

type Gemini struct {
	apiKey string
	model  string
	client *http.Client
}

func NewGemini(apiKey, model string) *Gemini {
	return &Gemini{apiKey, model, &http.Client{}}
}

func (g *Gemini) ChatStream(ctx context.Context, history []Message, systemPrompt string, onChunk func(string)) (string, Usage, error) {
	endpoint := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse",
		g.model,
	)

	// convert the request body to JSON
	jsondata, err := json.Marshal(buildRequest(history, systemPrompt))
	if err != nil {
		return "", Usage{}, err
	}

	// create a new HTTP request to the Gemini API
	response, err := g.doWithRetry(ctx, endpoint, jsondata)
	if err != nil {
		return "", Usage{}, err
	}

	defer response.Body.Close()

	// read the response body line by line and process each chunk of data
	var full strings.Builder
	var usage Usage

	// finishReason is used to capture the reason for the completion of the response, if any
	finishReason := ""

	// create a new scanner to read the response body line by line
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // allow long lines

	// iterate over each line in the response body
	for scanner.Scan() {

		// check if the line starts with "data: ", which indicates a chunk of data from the server
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue // blank separator lines and anything else
		}

		// unmarshal the JSON data from the line into a GeminiResponse struct
		var chunk GeminiResponse
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			return full.String(), usage, err
		}

		// update the usage statistics if available in the chunk
		if chunk.UsageMetadata.PromptTokenCount > 0 || chunk.UsageMetadata.CandidatesTokenCount > 0 {
			usage = Usage{
				InputTokens:  chunk.UsageMetadata.PromptTokenCount,
				OutputTokens: chunk.UsageMetadata.CandidatesTokenCount,
			}
		}
		if len(chunk.Candidates) == 0 {
			continue
		}

		cand := chunk.Candidates[0]
		if cand.FinishReason != "" {
			finishReason = cand.FinishReason
		}
		for _, p := range cand.Content.Parts {
			if p.Text != "" {
				full.WriteString(p.Text)
				onChunk(p.Text)
			}
		}
	}

	// check for any errors encountered while scanning the response body
	if err := scanner.Err(); err != nil {
		return full.String(), usage, err
	}

	// if no content was received, return an error indicating that Gemini returned no content
	if full.Len() == 0 {
		return "", usage, fmt.Errorf("gemini returned no content (finish reason: %s)", finishReason)
	}

	// return the full concatenated text, usage statistics, and nil error
	return full.String(), usage, nil
}

func buildRequest(history []Message, systemPrompt string) GeminiRequest {

	// create a slice of Content structs from the history of messages
	contents := make([]Content, 0, len(history))

	// iterate over the history of messages and convert them to Content structs
	for _, m := range history {
		role := "user"

		// if the message role is RoleAssistant, set the role to "model"
		if m.Role == RoleAssistant {
			role = "model"
		}

		// append the Content struct to the contents slice
		contents = append(contents, Content{
			Role:  role,
			Parts: []Part{{Text: m.Content}},
		})
	}

	// create a GeminiRequest struct with the contents and system prompt
	req := GeminiRequest{Contents: contents}
	if systemPrompt != "" {
		req.SystemInstruction = &Content{Parts: []Part{{Text: systemPrompt}}}
	}

	// return the GeminiRequest struct
	return req
}

// set a maximum number of attempts for retrying requests to the Gemini API
const maxAttempts = 4

// normalize the status code to determine if the request should be retried
func retryable(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

// doWithRetry sends the request and retries on temporary failures.
// It returns only a 200 response; the caller must close the body.
func (g *Gemini) doWithRetry(ctx context.Context, url string, body []byte) (*http.Response, error) {
	
	// lastErr is used to capture the last error encountered during the retry attempts
	var lastErr error

	// iterate over the maximum number of attempts to send the request
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			wait := time.Duration(1<<(attempt-1)) * time.Second // 1s, 2s, 4s
			select {
			case <-time.After(wait): 
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		// create a new HTTP request with the provided context, URL, and body
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-goog-api-key", g.apiKey)

		// send the request to the Gemini API and get the response
		resp, err := g.client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err() // cancelled or timed out: don't retry
			}
			lastErr = err // network error: retry
			continue
		}

		// check the response status code to determine if the request was successful or if it should be retried
		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		// read the response body and close it to avoid resource leaks
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		lastErr = fmt.Errorf("gemini api error (status %d): %s", resp.StatusCode, b)

		// if the status code is not retryable, return the last error encountered
		if !retryable(resp.StatusCode) {
			return nil, lastErr // 400/401/403/404: retrying can't help
		}
	}

	return nil, fmt.Errorf("gave up after %d attempts: %w", maxAttempts, lastErr)
}

func (g *Gemini) Chat(ctx context.Context, history []Message, systemPrompt string) (string, Usage, error) {
	return g.generate(ctx, buildRequest(history, systemPrompt))
}

func (g *Gemini) ChatJSON(ctx context.Context, history []Message, systemPrompt string, schema map[string]any) (string, Usage, error) {

	// build the GeminiRequest with the provided history, system prompt, and schema for JSON response
	req := buildRequest(history, systemPrompt)
	req.GenerationConfig = &GenerationConfig{
		ResponseMimeType: "application/json", // set the response MIME type to application/json
		ResponseSchema:   schema, // set the response schema for JSON validation
	}
	return g.generate(ctx, req)
}