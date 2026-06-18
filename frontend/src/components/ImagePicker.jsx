import { Select, Upload, Button } from "antd";
import { UploadOutlined } from "@ant-design/icons";

export default function ImagePicker({ cases, selected, onSelect, onUpload, disabled }) {
  return (
    <>
      <Select
        style={{ width: 220 }}
        placeholder="选择图片…"
        value={selected || undefined}
        onChange={onSelect}
        disabled={disabled}
        showSearch
        optionFilterProp="label"
        options={cases.map((c) => ({
          value: c.name,
          label: `${c.name}${c.has_result ? "  ✅" : ""}`,
        }))}
      />
      <Upload
        accept="image/*"
        showUploadList={false}
        disabled={disabled}
        beforeUpload={(file) => {
          onUpload(file);
          return false; // 阻止 antd 默认上传，由 onUpload 自行处理
        }}
      >
        <Button icon={<UploadOutlined />} disabled={disabled}>
          上传新图
        </Button>
      </Upload>
    </>
  );
}
