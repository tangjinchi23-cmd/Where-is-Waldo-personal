import { staticUrl } from "../api.js";

export default function CandidateGallery({ candidates }) {
  if (!candidates.length) return null;
  return (
    <div>
      <h3>检测候选（{candidates.length}）</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
        {candidates.map((c, i) => (
          <div
            key={i}
            style={{
              border: c.verified ? "3px solid #2ecc71" : "1px solid #ddd",
              padding: 4, borderRadius: 4, width: 140,
            }}
          >
            {c.crop_url && (
              <img src={staticUrl(c.crop_url)} alt={`候选 ${i}`} style={{ width: "100%", display: "block" }} />
            )}
            <small>
              #{i} conf={c.confidence != null ? c.confidence.toFixed(2) : "—"}
              {c.verified && " ✅"}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}
