import { useState } from "react";

export default function ImagePicker({ cases, selected, onSelect, onUpload, disabled }) {
  const [busy, setBusy] = useState(false);

  async function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setBusy(true);
    try {
      await onUpload(file);
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  }

  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
      <select value={selected || ""} onChange={(e) => onSelect(e.target.value)} disabled={disabled}>
        <option value="" disabled>选择图片…</option>
        {cases.map((c) => (
          <option key={c.name} value={c.name}>
            {c.name} {c.has_result ? "✅" : ""}
          </option>
        ))}
      </select>
      <label style={{ cursor: "pointer", color: "#2a6" }}>
        {busy ? "上传中…" : "上传新图"}
        <input type="file" accept="image/*" onChange={handleFile} disabled={disabled || busy} style={{ display: "none" }} />
      </label>
    </div>
  );
}
