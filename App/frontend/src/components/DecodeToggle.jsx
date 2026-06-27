import "./DecodeToggle.css";

const MODES = [
  {
    value: "autoregressive",
    label: "Autoregressive",
    icon: "",
    description: "Standard token-by-token decoding",
  },
  {
    value: "mtp_verified",
    label: "MTP Verified",
    icon: "",
    description: "Multi-Token Prediction with main-head verification",
  },
];

export default function DecodeToggle({ value, onChange, disabled }) {
  return (
    <div className="decode-toggle">
      <span className="decode-toggle__label"> Decode Mode</span>
      <div className="neo-toggle">
        {MODES.map((mode) => (
          <button
            key={mode.value}
            className={`neo-toggle__option ${
              value === mode.value ? "neo-toggle__option--active" : ""
            }`}
            onClick={() => onChange(mode.value)}
            disabled={disabled}
            type="button"
            title={mode.description}
          >
            <span>{mode.icon}</span> {mode.label}
          </button>
        ))}
      </div>
      <p className="decode-toggle__hint text-sm text-muted">
        {MODES.find((m) => m.value === value)?.description}
      </p>
    </div>
  );
}
