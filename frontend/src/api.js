// 后端通信封装。dev 下 /api、/static 由 vite 代理到 :8000。
const BASE = import.meta.env.VITE_API_BASE || "";

export async function fetchCases() {
  const r = await fetch(`${BASE}/api/cases`);
  if (!r.ok) throw new Error("加载图片列表失败");
  return r.json();
}

export async function uploadImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/api/upload`, { method: "POST", body: fd });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || "上传失败");
  }
  return r.json();
}

// 打开 SSE，逐事件回调；done/error 自动关闭。返回取消函数。
export function subscribeDetect(name, { onEvent, onError }) {
  const es = new EventSource(`${BASE}/api/detect?name=${encodeURIComponent(name)}`);
  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    onEvent(data);
    if (data.stage === "done" || data.stage === "error") es.close();
  };
  es.onerror = () => {
    es.close();
    onError && onError();
  };
  return () => es.close();
}

export function staticUrl(u) {
  return u ? `${BASE}${u}` : null;
}
