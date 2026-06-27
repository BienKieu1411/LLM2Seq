import MetricsBadge from "./MetricsBadge";
import "./ResultCard.css";

export default function ResultCard({ result, isLoading }) {
  if (isLoading) {
    return (
      <div className="result-card neo-card">
        <div className="result-card__header">
          <h2 className="result-card__title"> Summary</h2>
        </div>
        <div className="result-card__loading">
          <div className="neo-spinner" />
          <span>Generating summary…</span>
        </div>
        <div className="result-card__skeleton-lines">
          <div className="neo-skeleton" style={{ height: "1rem", width: "100%" }} />
          <div className="neo-skeleton" style={{ height: "1rem", width: "92%" }} />
          <div className="neo-skeleton" style={{ height: "1rem", width: "85%" }} />
          <div className="neo-skeleton" style={{ height: "1rem", width: "78%" }} />
        </div>
      </div>
    );
  }

  if (!result) return null;

  const { summary, decode_mode, latency_seconds, generated_tokens, tokens_per_second, mtp_metrics } = result;

  return (
    <div className="result-card neo-card">
      <div className="result-card__header">
        <h2 className="result-card__title"> Summary</h2>
        <span className={`neo-badge ${decode_mode === "mtp_verified" ? "neo-badge--purple" : "neo-badge--blue"}`}>
          {decode_mode === "mtp_verified" ? " MTP Verified" : " Autoregressive"}
        </span>
      </div>

      <div className="result-card__summary">
        <p>{summary}</p>
      </div>

      <hr className="neo-divider" />

      <div className="result-card__metrics-header">
        <h3> Performance Metrics</h3>
      </div>

      <div className="result-card__metrics">
        <MetricsBadge
          label="Latency"
          value={`${latency_seconds.toFixed(2)}s`}
          variant="latency"
        />
        <MetricsBadge
          label="Tokens"
          value={generated_tokens}
          variant="tokens"
        />
        <MetricsBadge
          label="Speed"
          value={`${tokens_per_second.toFixed(1)} tok/s`}
          variant="speed"
        />
        {mtp_metrics && mtp_metrics.acceptance_rate != null && (
          <MetricsBadge
            label="Acceptance"
            value={`${(mtp_metrics.acceptance_rate * 100).toFixed(1)}%`}
            variant="acceptance"
          />
        )}
        {mtp_metrics && mtp_metrics.num_steps != null && (
          <MetricsBadge
            label="Steps"
            value={mtp_metrics.num_steps}
            variant="steps"
          />
        )}
        {mtp_metrics && mtp_metrics.speedup_vs_autoregressive != null && (
          <MetricsBadge
            label="Speedup"
            value={`${mtp_metrics.speedup_vs_autoregressive.toFixed(2)}×`}
            variant="speed"
          />
        )}
      </div>
    </div>
  );
}
