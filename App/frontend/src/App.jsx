import { useState, useEffect, useCallback } from "react";
import TextInput from "./components/TextInput";
import DecodeToggle from "./components/DecodeToggle";
import ResultCard from "./components/ResultCard";
import { cleanWikiText } from "./utils/textCleaner";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [sourceText, setSourceText] = useState("");
  const [decodeMode, setDecodeMode] = useState("autoregressive");
  const [maxTokens, setMaxTokens] = useState(256);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        setModelReady(data.model_ready);

        if (data.model_ready) {
          const infoRes = await fetch(`${API_BASE}/api/model-info`);
          const infoData = await infoRes.json();
          setModelInfo(infoData);
        }
      } catch {
        setModelReady(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSummarize = useCallback(async () => {
    if (!sourceText.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: sourceText,
          decode_mode: decodeMode,
          max_new_tokens: maxTokens,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      
      const cleanedSummary = cleanWikiText(data.summary);

      setResult({
        ...data,
        summary: cleanedSummary
      });
    } catch (err) {
      setError(err.message || "Failed to generate summary");
    } finally {
      setIsLoading(false);
    }
  }, [sourceText, decodeMode, maxTokens, isLoading]);

  return (
    <div className="app">

      <header className="app__header">
        <div className="app__header-inner">
          <div className="app__logo">
            <h1 className="app__title">
              <span className="app__title-prefix">LLM</span>
              <span className="app__title-accent">2Seq</span>
            </h1>
            <span className="app__subtitle">Text Summarization</span>
          </div>
          <div className="app__status">
            <span
              className={`neo-badge ${modelReady ? "neo-badge--green" : "neo-badge--red"
                }`}
            >
              <span
                className={`app__status-dot ${modelReady ? "app__status-dot--online" : ""
                  }`}
              />
              {modelReady ? "Model Ready" : "Model Loading…"}
            </span>
            {modelInfo && (
              <span className="neo-badge neo-badge--yellow">
                {modelInfo.encoder_name.split("/").pop()}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="app__main">
        <div className="app__container">

          <section className="app__section neo-card" id="input-section">
            <TextInput
              value={sourceText}
              onChange={setSourceText}
              disabled={isLoading}
            />

            <div className="app__controls">
              <DecodeToggle
                value={decodeMode}
                onChange={setDecodeMode}
                disabled={isLoading}
              />

              <div className="app__tokens-control">
                <label className="app__tokens-label" htmlFor="max-tokens">
                  Max Tokens: <strong>{maxTokens}</strong>
                </label>
                <input
                  id="max-tokens"
                  type="range"
                  min={32}
                  max={512}
                  step={16}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  disabled={isLoading}
                  className="app__slider"
                />
              </div>
            </div>

            <button
              className="neo-btn neo-btn--primary app__submit-btn"
              onClick={handleSummarize}
              disabled={isLoading || !sourceText.trim() || !modelReady}
              type="button"
              id="summarize-button"
            >
              {isLoading ? (
                <>
                  <span className="neo-spinner" /> Generating…
                </>
              ) : (
                <> Summarize</>
              )}
            </button>

            {!modelReady && (
              <p className="app__warning text-sm">
                Model is still loading. Please wait…
              </p>
            )}
          </section>

          {error && (
            <div className="app__error neo-card" id="error-section">
              <strong> Error:</strong> {error}
            </div>
          )}

          <ResultCard result={result} isLoading={isLoading} />


        </div>
      </main>

      <footer className="app__footer">
        <p>
          <strong>LLM2Seq</strong> — Encoder-Decoder with Multi-Token Prediction
          {" · "}
          <a
            href="https://github.com/BienKieu1411/LLM2Seq"
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem" }}
          >
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 19 19">
              <use href="/icons.svg#github-icon" />
            </svg>
            GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
