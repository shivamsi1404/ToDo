package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

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
	systemPrompt := "You are ToDo. Act like my personal assistant reply in a helpful and informative way but limited in length. Treat me as your boss and stay as my main guide. You are a helpful assistant that provides information and answers questions. You should be concise and to the point, but also provide enough detail to be helpful. You should also be polite and respectful. You should provide personal opinions or advice but with proper facts."

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

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)

		fmt.Print("ai> ")
		reply, usage, err := provider.ChatStream(ctx, history, systemPrompt, func(s string) {
			fmt.Print(s)
		})
		cancel() // after the call returns, not inside the callback
		fmt.Println()

		if err != nil {
			fmt.Println("error:", err)
			history = history[:len(history)-1] // drop the failed turn
			continue
		}

		history = append(history, llm.Message{Role: llm.RoleAssistant, Content: reply})
		fmt.Printf("[tokens: in=%d out=%d]\n", usage.InputTokens, usage.OutputTokens)
	}
	fmt.Println("Chat ended.")
}
