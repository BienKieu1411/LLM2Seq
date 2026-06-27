import "./MetricsBadge.css";

const VARIANT_MAP = {
  latency: { color: "blue", icon: "" },
  tokens: { color: "green", icon: "" },
  speed: { color: "purple", icon: "" },
  acceptance: { color: "orange", icon: "" },
  steps: { color: "yellow", icon: "" },
  default: { color: "blue", icon: "" },
};

export default function MetricsBadge({ label, value, variant = "default" }) {
  const { color, icon } = VARIANT_MAP[variant] || VARIANT_MAP.default;

  return (
    <div className={`metrics-badge neo-badge neo-badge--${color}`}>
      <span className="metrics-badge__icon">{icon}</span>
      <span className="metrics-badge__content">
        <span className="metrics-badge__label">{label}</span>
        <span className="metrics-badge__value">{value}</span>
      </span>
    </div>
  );
}
