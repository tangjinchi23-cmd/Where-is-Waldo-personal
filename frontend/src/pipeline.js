// 纯函数事件归约器：把 SSE 事件序列折叠成 UI 状态。无副作用，便于单测。

export const INITIAL = {
  stages: { segment: "pending", detect: "pending", verify: "pending", done: "pending" },
  patches: null,
  count: null,
  candidates: [],
  verify: null,
  result: null,
  error: null,
};

export function applyEvent(state, ev) {
  const s = { ...state, stages: { ...state.stages } };
  switch (ev.stage) {
    case "segment":
      s.stages.segment = "done";
      s.stages.detect = "running";
      s.patches = ev.patches;
      break;
    case "detect":
      s.stages.detect = "done";
      s.stages.verify = "running";
      s.count = ev.count;
      s.candidates = ev.candidates || [];
      break;
    case "verify":
      s.stages.verify = "done";
      s.stages.done = "running";
      s.verify = { ran: ev.ran, choice: ev.choice ?? -1 };
      if (ev.candidates) s.candidates = ev.candidates;
      break;
    case "done":
      s.stages.done = "done";
      s.result = {
        found: ev.found,
        verifyRan: ev.verify_ran,
        bbox: ev.bbox,
        resultUrl: ev.result_url,
      };
      break;
    case "error":
      s.error = ev.message;
      for (const k of Object.keys(s.stages)) {
        if (s.stages[k] === "running") s.stages[k] = "error";
      }
      break;
    default:
      break;
  }
  return s;
}
