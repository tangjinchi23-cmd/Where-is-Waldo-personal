import { Card, Image, Tag, Empty } from "antd";
import { staticUrl } from "../api.js";

export default function CandidateGallery({ candidates }) {
  return (
    <Card title={`检测候选${candidates.length ? `（${candidates.length}）` : ""}`} size="small">
      {!candidates.length ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无候选" />
      ) : (
        <Image.PreviewGroup>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
              gap: 12,
            }}
          >
            {candidates.map((c, i) => (
              <div
                key={i}
                style={{
                  border: c.verified ? "2px solid #52c41a" : "1px solid #f0f0f0",
                  borderRadius: 8,
                  padding: 6,
                }}
              >
                {c.crop_url && (
                  <Image
                    src={staticUrl(c.crop_url)}
                    alt={`候选 ${i}`}
                    style={{ borderRadius: 4 }}
                  />
                )}
                <div
                  style={{
                    marginTop: 6,
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <span style={{ fontSize: 12, color: "#999" }}>#{i}</span>
                  {c.verified ? (
                    <Tag color="success">Waldo</Tag>
                  ) : (
                    <Tag>conf {c.confidence != null ? c.confidence.toFixed(2) : "—"}</Tag>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Image.PreviewGroup>
      )}
    </Card>
  );
}
