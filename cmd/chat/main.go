package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"

	"github.com/shivamsi1404/ToDo/internal/llm"
)

func main() {
	_ = godotenv.Load() // fine if .env is missing and the key is already in the environment

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("GEMINI_API_KEY is not set")
		os.Exit(1)
	}

	var provider llm.Provider = llm.NewGemini(apiKey, "gemini-3.5-flash-lite")
	systemPrompt := "Answer in one short sentence."

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

		history = append(history, llm.Message{Role: llm.RoleUser, Content: line})

		reply, usage, err := provider.Chat(context.Background(), history, systemPrompt)
		if err != nil {
			fmt.Println("error:", err)
			history = history[:len(history)-1] // drop the failed turn
			continue
		}

		history = append(history, llm.Message{Role: llm.RoleAssistant, Content: reply})
		fmt.Printf("ai> %s\n[tokens: in=%d out=%d]\n", reply, usage.InputTokens, usage.OutputTokens)
	}
}