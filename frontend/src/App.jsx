import { useEffect, useState } from "react";
import { fetchCases, uploadImage, subscribeDetect } from "./api.js";
import { INITIAL, applyEvent } from "./pipeline.js";
import ImagePicker from "./components/ImagePicker.jsx";
import PipelineProgress from "./components/PipelineProgress.jsx";
import CandidateGallery from "./components/CandidateGallery.jsx";
import ResultView from "./components/ResultView.jsx";

export default function App() {
  const [cases, setCases] = useState([]);
  const [selected, setSelected] = useState("");
  const [imageUrl, setImageUrl] = useState(null);
  const [pipe, setPipe] = useState(INITIAL);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetchCases().then(setCases).catch((e) => console.error(e));
  }, []);

  function pickCase(name) {
    setSelected(name);
    const c = cases.find((x) => x.name === name);
    setImageUrl(c ? c.image_url : null);
    setPipe(INITIAL);
  }

  async function handleUpload(file) {
    const { name, image_url } = await uploadImage(file);
    setCases((prev) => [...prev.filter((c) => c.name !== name), { name, image_url, has_result: false }]);
    setSelected(name);
    setImageUrl(image_url);
    setPipe(INITIAL);
  }

  function runDetect() {
    if (!selected) return;
    setPipe(INITIAL);
    setRunning(true);
    subscribeDetect(selected, {
      onEvent: (ev) => {
        setPipe((prev) => applyEvent(prev, ev));
        if (ev.stage === "done" || ev.stage === "error") setRunning(false);
      },
      onError: () => {
        setPipe((prev) => applyEvent(prev, { stage: "error", message: "连接中断" }));
        setRunning(false);
      },
    });
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>Where's Waldo Agent</h1>
      <ImagePicker
        cases={cases}
        selected={selected}
        onSelect={pickCase}
        onUpload={handleUpload}
        disabled={running}
      />
      <button onClick={runDetect} disabled={!selected || running} style={{ marginTop: 12 }}>
        {running ? "检测中…" : "运行检测"}
      </button>

      <PipelineProgress stages={pipe.stages} patches={pipe.patches} count={pipe.count} verify={pipe.verify} />
      {pipe.error && <p style={{ color: "#e74c3c" }}>错误：{pipe.error}</p>}

      <CandidateGallery candidates={pipe.candidates} />
      <ResultView imageUrl={imageUrl} result={pipe.result} />
    </div>
  );
}
