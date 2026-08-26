// Command ara-webui is the public-facing UI and API gateway for ARA-1.
//
// WHY GO IN FRONT OF PYTHON, NOT A REWRITE:
// The agent itself (LangGraph ReAct loop, synthesis DAG, yfinance/EDGAR
// tools) stays in Python — that logic is the whole project and rewriting it
// here would just be risk with no payoff. Go's job is the part that
// benefits from it: a small, fast, dependency-free HTTP server serving the
// static UI and proxying analysis requests to the existing FastAPI backend
// (api/server.py) with proper timeouts and graceful shutdown, replacing
// Streamlit as the thing people actually load in a browser.
//
// Run:
//
//	go run . -addr :8080 -python-api http://localhost:8000
//
// The Python side must be running separately:
//
//	.venv/Scripts/python.exe -m uvicorn api.server:app --port 8000
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	addr := flag.String("addr", envOr("ARA_WEBUI_ADDR", ":8080"), "address to listen on")
	pythonAPI := flag.String("python-api", envOr("ARA_PYTHON_API", "http://localhost:8000"), "base URL of the Python FastAPI backend")
	staticDir := flag.String("static", envOr("ARA_STATIC_DIR", "static"), "directory to serve the frontend from")
	flag.Parse()

	proxy := &backendProxy{
		baseURL: *pythonAPI,
		// Analysis can take a couple of minutes (LLM tool-calling loop +
		// synthesis); health/report calls are cheap. Two clients, two
		// timeouts, rather than one timeout wrong for both.
		fastClient: &http.Client{Timeout: 10 * time.Second},
		longClient: &http.Client{Timeout: 5 * time.Minute},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", proxy.handleHealth)
	mux.HandleFunc("POST /api/analyze", proxy.handleAnalyze)
	mux.HandleFunc("GET /api/reports", proxy.handleListReports)
	mux.HandleFunc("GET /api/reports/{filename}", proxy.handleGetReport)
	mux.Handle("/", http.FileServer(http.Dir(*staticDir)))

	handler := withLogging(withRecover(mux))

	server := &http.Server{
		Addr:         *addr,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 6 * time.Minute, // must exceed longClient's timeout
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown: stop accepting new requests on SIGINT/SIGTERM but
	// let in-flight analyses finish rather than cutting them off.
	go func() {
		log.Printf("ara-webui listening on %s (proxying to %s, serving %s)", *addr, *pythonAPI, *staticDir)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop

	log.Println("shutting down, waiting for in-flight requests...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("shutdown error: %v", err)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
