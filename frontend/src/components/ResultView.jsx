import { Card, Image, Tag, Empty, Space, Flex } from "antd";
import { staticUrl } from "../api.js";

function Labeled({ label, children }) {
  return (
    <div>
      <div style={{ marginBottom: 6, color: "#999", fontSize: 12 }}>{label}</div>
      {children}
    </div>
  );
}

export default function ResultView({ imageUrl, result }) {
  return (
    <Card title="原图 / 结果" size="small">
      <Space size="large" align="start" wrap>
        <Labeled label="原图">
          {imageUrl ? (
            <Image src={staticUrl(imageUrl)} width={360} />
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="未选择" />
          )}
        </Labeled>

        <Labeled label="结果">
          {!result && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="尚未检测" />}
          {result && !result.found && <Tag color="error">未找到 Waldo</Tag>}
          {result && result.found && (
            <Flex vertical gap="small">
              {result.resultUrl ? (
                <Image src={staticUrl(result.resultUrl)} width={360} />
              ) : (
                <Tag>结果图未生成</Tag>
              )}
              <Space wrap>
                <Tag color="blue">bbox {JSON.stringify(result.bbox)}</Tag>
                <Tag color={result.verifyRan ? "green" : "gold"}>
                  {result.verifyRan ? "verify 确认" : "detect 单候选"}
                </Tag>
              </Space>
            </Flex>
          )}
        </Labeled>
      </Space>
    </Card>
  );
}
