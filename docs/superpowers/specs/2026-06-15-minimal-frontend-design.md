# 最小前端 + 服务层设计

- **日期**：2026-06-15
- **分支**：`frontend`
- **目标**：为 WhereisWaldoAgent 创建一个最小、便于改动的前端，用于预览已有的 Waldo 检测结果；同时作为验证团队 GitHub 协作流程（开分支 → push → PR → 合并）的载体。
- **状态**：设计已确认，待编写实现计划。

---

## 1. 背景与约束

- 入口 `run_agent(image_path, grid_size=1)` 返回 `WaldoState`，副作用是把标注图保存到 `outputs/{basename}_result.jpg`。
- agent 跑一次又慢又烧 API 钱（analyze/detect/verify 全是 gpt-5.5 推理模型调用）。
- 当前 `outputs/` 只存结果图，**未持久化 bbox 数字**（`verified_result` 不落盘）。
- 项目最终会引入 **FastAPI + React** 等正式框架，本期前端需为此铺路、避免推倒重来。

## 2. 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 前端形态 | **Streamlit** | 纯 Python，和现有项目零摩擦，最易改 |
| 分层 | **Streamlit + 独立 service 层** | service 包装数据契约，UI 换皮（React）时 service 不动 |
| 交互范围 | **仅预览已有结果** | 不烧 API 钱、秒开；实时检测留给将来 FastAPI 版 |
| service 位置 | **与 `frontend/` 平级**（独立 `service/`） | service 是后端、将来被 FastAPI 复用，不能塞进会变成 React 的 `frontend/` 里 |

## 3. 文件结构

```
WhereisWaldoAgent/
├── service/                      # 共享后端服务层（将来 FastAPI 复用）
│   ├── __init__.py
│   └── waldo_service.py          # 数据契约 + 查询函数
├── frontend/                     # UI 层（将来换成 React）
│   ├── README.md                 # 如何运行
│   ├── requirements.txt          # streamlit
│   └── app.py                    # Streamlit 画廊页面
└── tests/
    └── test_waldo_service.py     # service 层单元测试
```

过渡路径：将来 `service/` 不动 → 新增 `api/`（FastAPI 调 service）→ `frontend/` 换 React 调 API。

## 4. service 层契约（最该稳定的部分）

```python
from dataclasses import dataclass

@dataclass
class WaldoCase:
    name: str                 # "1"
    image_path: str           # original-images/1.jpg
    result_path: str | None   # outputs/1_result.jpg（不存在则 None）
    has_result: bool

def list_cases() -> list[WaldoCase]:
    """扫描 original-images/ 下的图片，与 outputs/{name}_result.jpg 配对。
    目录不存在时返回空列表，不抛异常。按 name 排序。"""

def get_case(name: str) -> WaldoCase | None:
    """按 name 返回单个 case；原图不存在返回 None。"""
```

这是前端唯一依赖的接口。将来 FastAPI 把这两个函数包成 `GET /cases`、`GET /cases/{name}`，React 直接调，service 一行不改。

实现要点：
- 原图目录常量指向项目根的 `original-images/`，结果目录指向 `outputs/`，用相对项目根的绝对路径解析，避免受 Streamlit 工作目录影响。
- 支持的图片后缀：`.jpg`、`.jpeg`、`.png`。
- `name` = 原图文件名去后缀（如 `1.jpg` → `1`）。
- 结果配对规则：`outputs/{name}_result.jpg` 存在即 `has_result=True`。

## 5. Streamlit 页面（app.py）

- 启动时调 `list_cases()`，侧边栏下拉/列表选一张图（标注哪些已有结果）。
- 选中后主区域并排显示：**原图** + **标注结果图**（红框）。
- 无结果的图 → 显示原图 + 提示"尚未检测"。
- 页面顶部一句说明：本页只预览已有结果，实时检测将在 FastAPI 版接入。

## 6. 错误处理

- service 对缺失目录/文件防御：目录不存在 → 返回空列表；`get_case` 原图缺失 → 返回 None。
- app.py 对空 case 列表显示友好提示，不崩。

## 7. 测试策略

- **TDD**：先为 `service/waldo_service.py` 写单元测试再实现。
  - `list_cases` 在临时目录下返回正确配对、排序、空目录返回空列表。
  - `get_case` 命中 / 未命中 / has_result 正确。
- Streamlit UI 不写自动化测试，靠手动 `streamlit run frontend/app.py` 冒烟验证。

## 8. 明确不做（YAGNI）

- 不在本期实时调用 `run_agent`（留给 FastAPI 版）。
- 不持久化 bbox 数字 / 不显示坐标数值（需先加 `results.json` 落盘，本期不做）。
- 不引入 FastAPI / React（本期只搭 service + Streamlit）。