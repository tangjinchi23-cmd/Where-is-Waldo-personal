const LABELS = { segment: "切片", detect: "检测", verify: "验证", done: "完成" };
const COLOR = { pending: "#ccc", running: "#f5a623", done: "#2ecc71", error: "#e74c3c" };

export default function PipelineProgress({ stages, patches, count, verify }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "12px 0" }}>
      {Object.keys(LABELS).map((k) => (
        <div key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 12, height: 12, borderRadius: "50%", background: COLOR[stages[k]] }} />
          <span>{LABELS[k]}</span>
          {k === "segment" && patches != null && <small>({patches} patch)</small>}
          {k === "detect" && count != null && <small>({count} 候选)</small>}
          {k === "verify" && verify && <small>({verify.ran ? `选中 #${verify.choice}` : "跳过"})</small>}
        </div>
      ))}
    </div>
  );
}
