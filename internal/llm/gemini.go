package llm

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "time"

    "github.com/joho/godotenv"
)

type GeminiRequest struct {
	Contents []Content `json:"contents"`
}

type Content struct {
	Parts []Part `json:"parts"`
}

type Part struct {
	Text string `json:"text"`
}

type GeminiResponse struct {
	Candidates []Candidate `json:"candidates"`
}

type Candidate struct {
	Content Content `json:"content"`
}

func GenerateGeminiResponse(prompt string) (string, error) {

    // load the environment variables from the .env file
    err := godotenv.Load()
    if err != nil {
        return "", err
    }

    // got the api key from the environment variable
	apiKey := os.Getenv("GEMINI_API_KEY")

    // fmt.Println("API Key:", apiKey) // Debugging line to check if the API key is being read correctly
    
    // get the endpoint for the Gemini API
    endpoint := "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"

    // generate the request body for the Gemini API
    requestbody := GeminiRequest{
        Contents: []Content{
            {
                Parts: []Part{
                    {
                        Text: prompt,
                    },
                },
            },
        },
    }

    // convert the request body to JSON
    jsondata, err := json.Marshal(requestbody)
    if err != nil {
        return "", err
    }

    // crete a new HTTP request to the Gemini API
    request,err := http.NewRequest("POST", endpoint, bytes.NewBuffer(jsondata))
    if err != nil {
        return "", err
    }

    // set the headers for the request
    request.Header.Set("Content-Type", "application/json")
    request.Header.Set("x-goog-api-key",apiKey)

    // create a new HTTP client with a timeout
    client := &http.Client{
        Timeout: 30 * time.Second,
    }

    // get the response from the Gemini API
    response, err := client.Do(request)
    if err != nil {
        return "", err
    }
    defer response.Body.Close()

    // read the response body and check for errors
    if response.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(response.Body)
        return "", fmt.Errorf("Gemini API error: %s", body)
    }

    // decode the response body into a GeminiResponse struct
    var result GeminiResponse

    err = json.NewDecoder(response.Body).Decode(&result)
    if err != nil {
        return "", err
    }

    // check if result.Candidates and result.Candidates[0].Content.Parts are not empty
    if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
        return "", fmt.Errorf("Gemini API returned no content")
    }

    return result.Candidates[0].Content.Parts[0].Text, nil
}
