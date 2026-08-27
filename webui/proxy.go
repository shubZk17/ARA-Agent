package main

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
)

// backendProxy forwards UI requests to the Python FastAPI backend
// (api/server.py). It never trusts the backend to be up: every handler
// turns a connection failure into a clean JSON error instead of a hang or
// a raw Go stack trace, since the person on the other end is a browser tab,
// not a developer reading logs.
type backendProxy struct {
	baseURL    string
	fastClient *http.Client
	longClient *http.Client
}

const maxAnalyzeBody = 1 << 16 // 64KB — a research query is never bigger than this

type errorResponse struct {
	Error string `json:"error"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, errorResponse{Error: msg})
}

// handleHealth reports both this gateway's own health and whether it can
// currently reach the Python analysis backend — a UI showing "healthy" when
// the thing it depends on is down is worse than no health check at all.
func (p *backendProxy) handleHealth(w http.ResponseWriter, r *http.Request) {
	type health struct {
		Status    string `json:"status"`
		Backend   string `json:"backend"`
		BackendUp bool   `json:"backend_up"`
	}

	resp, err := p.fastClient.Get(p.baseURL + "/health")
	backendUp := err == nil && resp != nil && resp.StatusCode == http.StatusOK
	if resp != nil {
		resp.Body.Close()
	}

	writeJSON(w, http.StatusOK, health{
		Status:    "healthy",
		Backend:   p.baseURL,
		BackendUp: backendUp,
	})
}

// handleAnalyze proxies POST /api/analyze straight to the Python /analyze
// endpoint. The body is passed through as-is (query, horizon, risk_profile,
// max_iterations, enable_evaluation) — the contract lives in
// api/server.py's AnalysisRequest, one source of truth, not duplicated here.
func (p *backendProxy) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxAnalyzeBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "request body too large or unreadable")
		return
	}

	req, err := http.NewRequest(http.MethodPost, p.baseURL+"/analyze", strings.NewReader(string(body)))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to build upstream request")
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.longClient.Do(req)
	if err != nil {
		if errors.Is(err, http.ErrHandlerTimeout) || isTimeout(err) {
			writeError(w, http.StatusGatewayTimeout,
				"analysis took too long (over 5 minutes) — the LLM provider may be rate-limited or unresponsive")
			return
		}
		writeError(w, http.StatusBadGateway,
			"could not reach the analysis backend at "+p.baseURL+" — is `uvicorn api.server:app` running?")
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// handleEvaluate proxies POST /api/evaluate — same body-passthrough,
// long-timeout shape as handleAnalyze, since it also runs the full agent.
func (p *backendProxy) handleEvaluate(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxAnalyzeBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "request body too large or unreadable")
		return
	}

	req, err := http.NewRequest(http.MethodPost, p.baseURL+"/evaluate", strings.NewReader(string(body)))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to build upstream request")
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.longClient.Do(req)
	if err != nil {
		if errors.Is(err, http.ErrHandlerTimeout) || isTimeout(err) {
			writeError(w, http.StatusGatewayTimeout,
				"evaluation took too long (over 5 minutes) — the LLM provider may be rate-limited or unresponsive")
			return
		}
		writeError(w, http.StatusBadGateway,
			"could not reach the analysis backend at "+p.baseURL+" — is `uvicorn api.server:app` running?")
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// handleConfig proxies GET /api/config — same shape as handleListReports.
func (p *backendProxy) handleConfig(w http.ResponseWriter, r *http.Request) {
	resp, err := p.fastClient.Get(p.baseURL + "/config")
	if err != nil {
		writeError(w, http.StatusBadGateway, "could not reach the analysis backend")
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (p *backendProxy) handleListReports(w http.ResponseWriter, r *http.Request) {
	resp, err := p.fastClient.Get(p.baseURL + "/reports")
	if err != nil {
		writeError(w, http.StatusBadGateway, "could not reach the analysis backend")
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// handleGetReport streams one report file through. filename comes from the
// URL path, so it's re-validated here even though api/server.py already
// resolve-and-contains it (D11) — a gateway should not blindly trust that
// every request it forwards was already sanitized downstream.
func (p *backendProxy) handleGetReport(w http.ResponseWriter, r *http.Request) {
	filename := r.PathValue("filename")
	if filename == "" || strings.Contains(filename, "/") || strings.Contains(filename, "..") {
		writeError(w, http.StatusBadRequest, "invalid filename")
		return
	}

	resp, err := p.fastClient.Get(p.baseURL + "/reports/" + filename)
	if err != nil {
		writeError(w, http.StatusBadGateway, "could not reach the analysis backend")
		return
	}
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func isTimeout(err error) bool {
	type timeoutError interface{ Timeout() bool }
	var te timeoutError
	if errors.As(err, &te) {
		return te.Timeout()
	}
	return false
}
