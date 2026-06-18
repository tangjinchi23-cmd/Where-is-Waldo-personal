import { useEffect, useState } from "react";
import { Layout, Card, Button, Space, Row, Col, Alert, App as AntdApp } from "antd";
import { PlayCircleOutlined } from "@ant-design/icons";
import { fetchCases, uploadImage, subscribeDetect } from "./api.js";
import { INITIAL, applyEvent } from "./pipeline.js";
import ImagePicker from "./components/ImagePicker.jsx";
import PipelineProgress from "./components/PipelineProgress.jsx";
import CandidateGallery from "./components/CandidateGallery.jsx";
import ResultView from "./components/ResultView.jsx";

const { Header, Content } = Layout;

export default function App() {
  const { message } = AntdApp.useApp();
  const [cases, setCases] = useState([]);
  const [selected, setSelected] = useState("");
  const [imageUrl, setImageUrl] = useState(null);
  const [pipe, setPipe] = useState(INITIAL);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetchCases()
      .then(setCases)
      .catch((e) => message.error(e.message || "加载图片列表失败"));
  }, [message]);

  function pickCase(name) {
    setSelected(name);
    const c = cases.find((x) => x.name === name);
    setImageUrl(c ? c.image_url : null);
    setPipe(INITIAL);
  }

  async function handleUpload(file) {
    try {
      const { name, image_url } = await uploadImage(file);
      setCases((prev) => [...prev.filter((c) => c.name !== name), { name, image_url, has_result: false }]);
      setSelected(name);
      setImageUrl(image_url);
      setPipe(INITIAL);
      message.success(`已上传 ${name}`);
    } catch (e) {
      message.error(e.message || "上传失败");
    }
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
    <Layout style={{ minHeight: "100vh" }}>
      <Header style={{ display: "flex", alignItems: "center" }}>
        <span style={{ color: "#fff", fontSize: 18, fontWeight: 600 }}>
          Where&apos;s Waldo Agent
        </span>
      </Header>
      <Content style={{ padding: 24, maxWidth: 1280, margin: "0 auto", width: "100%" }}>
        <Space style={{ marginBottom: 16 }} wrap>
          <ImagePicker
            cases={cases}
            selected={selected}
            onSelect={pickCase}
            onUpload={handleUpload}
            disabled={running}
          />
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={runDetect}
            disabled={!selected}
            loading={running}
          >
            {running ? "检测中…" : "运行检测"}
          </Button>
        </Space>

        <Card size="small" style={{ marginBottom: 16 }}>
          <PipelineProgress
            stages={pipe.stages}
            patches={pipe.patches}
            count={pipe.count}
            verify={pipe.verify}
          />
        </Card>

        {pipe.error && (
          <Alert
            type="error"
            showIcon
            message={`错误：${pipe.error}`}
            style={{ marginBottom: 16 }}
          />
        )}

        <Row gutter={16}>
          <Col xs={24} lg={10}>
            <CandidateGallery candidates={pipe.candidates} />
          </Col>
          <Col xs={24} lg={14}>
            <ResultView imageUrl={imageUrl} result={pipe.result} />
          </Col>
        </Row>
      </Content>
    </Layout>
  );
}
