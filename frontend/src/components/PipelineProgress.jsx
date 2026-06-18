import { Steps } from "antd";

const ORDER = ["segment", "detect", "verify", "done"];
const LABELS = { segment: "切片", detect: "检测", verify: "验证", done: "完成" };
const STATUS = { pending: "wait", running: "process", done: "finish", error: "error" };

export default function PipelineProgress({ stages, patches, count, verify }) {
  const items = ORDER.map((k) => {
    let description;
    if (k === "segment" && patches != null) description = `${patches} patch`;
    if (k === "detect" && count != null) description = `${count} 候选`;
    if (k === "verify" && verify) description = verify.ran ? `选中 #${verify.choice}` : "跳过";
    return { title: LABELS[k], status: STATUS[stages[k]], description };
  });

  // current = 最后一个非 pending 的节点
  let current = 0;
  ORDER.forEach((k, i) => {
    if (stages[k] !== "pending") current = i;
  });

  return <Steps current={current} items={items} />;
}
