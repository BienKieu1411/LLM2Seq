import { useState, useRef, useEffect } from "react";
import { cleanWikiText } from "../utils/textCleaner";
import "./TextInput.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export default function TextInput({ value, onChange, disabled }) {
  const [charCount, setCharCount] = useState(0);
  const [isFetching, setIsFetching] = useState(false);
  const textareaRef = useRef(null);

  useEffect(() => {
    setCharCount(value.length);
  }, [value]);

  const handleRandomSample = async () => {
    try {
      setIsFetching(true);
      const res = await fetch(`${API_BASE}/api/random-sample`);
      if (!res.ok) throw new Error("Failed to fetch sample");
      const data = await res.json();
      const cleanedSource = cleanWikiText(data.source);
      onChange(cleanedSource);
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    } catch (err) {
      console.error(err);
      alert("Could not load sample: " + err.message);
    } finally {
      setIsFetching(false);
    }
  };

  const handleClear = () => {
    onChange("");
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  return (
    <div className="text-input">
      <div className="text-input__header">
        <label className="text-input__label" htmlFor="source-text">
           Source Text
        </label>
        <div className="text-input__actions">
          <button
            className="neo-btn neo-btn--small neo-btn--secondary"
            onClick={handleRandomSample}
            disabled={disabled || isFetching}
            type="button"
          >
            {isFetching ? "Loading..." : "WikiLingua Test Sample"}
          </button>
          {value.length > 0 && (
            <button
              className="neo-btn neo-btn--small neo-btn--secondary"
              onClick={handleClear}
              disabled={disabled}
              type="button"
            >
              ✕ Clear
            </button>
          )}
        </div>
      </div>

      <textarea
        ref={textareaRef}
        id="source-text"
        className="neo-input text-input__textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Paste or type the text you want to summarise…"
        disabled={disabled}
        rows={8}
      />

      <div className="text-input__footer">
        <span className="text-sm text-muted">
          {charCount.toLocaleString()} characters
          {" · "}
          {value.split(/\s+/).filter(Boolean).length.toLocaleString()} words
        </span>
      </div>
    </div>
  );
}
