package llm

import (
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
func (g *Gemini) Chat(ctx context.Context, history []Message, systemPrompt string) (string, Usage, error) {

	// get the endpoint for the Gemini API
	endpoint := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent",
		g.model,
	)

	// create a new GeminiRequest struct with the system prompt and history
	contents := make([]Content, 0, len(history)) // initialize the contents slice with the length of history
	// iterate over the history and create a Content struct for each message
	for _, m := range history {
		role := "user"
		if m.Role == RoleAssistant {
			role = "model"
		}
		contents = append(contents, Content{
			Role:  role,
			Parts: []Part{{Text: m.Content}},
		})
	}

	// create the request body for the Gemini API
	requestbody := GeminiRequest{Contents: contents}

	// if the system prompt is not empty, add it to the request body
	if systemPrompt != "" {
		requestbody.SystemInstruction = &Content{
			Parts: []Part{{Text: systemPrompt}},
		}
	}

	// convert the request body to JSON
	jsondata, err := json.Marshal(requestbody)
	if err != nil {
		return "", Usage{}, err
	}

	// crete a new HTTP request to the Gemini API
	request, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(jsondata))
	if err != nil {
		return "", Usage{}, err
	}

	// set the headers for the request
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("x-goog-api-key", g.apiKey)

	// send the request to the Gemini API and get the response`
	response, err := g.client.Do(request)
	if err != nil {
		return "", Usage{}, err
	}

	// ensure the response body is closed after reading
	defer response.Body.Close()

	// read the response body and check for errors
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		return "", Usage{}, fmt.Errorf("gemini api error (status %d): %s", response.StatusCode, body)
	}

	// read the response body raw and print it for debugging purposes
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return "", Usage{}, err
	}
	fmt.Println("Raw response from Gemini API:", string(raw))

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
	return &Gemini{apiKey, model, &http.Client{Timeout: 30 * time.Second}}
}
