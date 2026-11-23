package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestReadNoteRejectsTraversal(t *testing.T) {
	for _, bad := range []string{"../.env", "../../etc/passwd", "a/b", "/etc/passwd", "..", "", "sub/../x"} {
		if _, err := readNote(context.Background(), map[string]any{"name": bad}); err == nil {
			t.Errorf("name %q should have been rejected", bad)
		}
	}
}

func TestReadNoteWorks(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "notes"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes", "hello.md"), []byte("hi there"), 0o644); err != nil {
		t.Fatal(err)
	}

	// notesDir is relative, so run the test from the temp folder
	old, _ := os.Getwd()
	defer os.Chdir(old)
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}

	out, err := readNote(context.Background(), map[string]any{"name": "hello"})
	if err != nil || out["content"] != "hi there" {
		t.Fatalf("got %v, %v", out, err)
	}
}