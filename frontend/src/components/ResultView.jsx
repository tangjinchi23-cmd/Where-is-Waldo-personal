import { staticUrl } from "../api.js";

export default function ResultView({ imageUrl, result }) {
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
      <div>
        <h3>原图</h3>
        {imageUrl && <img src={staticUrl(imageUrl)} alt="原图" style={{ maxWidth: 480, display: "block" }} />}
      </div>
      <div>
        <h3>结果</h3>
        {!result && <p>尚未检测。</p>}
        {result && !result.found && <p>未找到 Waldo。</p>}
        {result && result.found && (
          <>
            {result.resultUrl ? (
              <img src={staticUrl(result.resultUrl)} alt="结果" style={{ maxWidth: 480, display: "block" }} />
            ) : (
              <p>（结果图未生成）</p>
            )}
            <p>
              bbox: {JSON.stringify(result.bbox)}
              {result.verifyRan ? "（verify 确认）" : "（detect 单候选，跳过 verify）"}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
