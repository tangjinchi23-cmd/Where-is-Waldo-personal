# 最小前端 + 服务层 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `frontend` 分支上搭建一个仅预览已有 Waldo 检测结果的 Streamlit 页面，并配一个与之解耦的共享 service 层，为将来 FastAPI + React 铺路。

**架构：** `service/waldo_service.py` 扫描 `original-images/` 与 `outputs/` 配对成 `WaldoCase` 列表（纯函数、易测）；`frontend/app.py`（Streamlit）只调 service 展示原图 + 结果图。service 与 frontend 平级解耦，将来 FastAPI 复用 service、React 接管 frontend。

**技术栈：** Python 3.10、Streamlit、pytest、dataclass、pathlib。

参考规格：`docs/superpowers/specs/2026-06-15-minimal-frontend-design.md`

---

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|------|
| `service/__init__.py` | 标记 package | 创建（空） |
| `service/waldo_service.py` | 数据契约 `WaldoCase` + `list_cases` / `get_case` | 创建 |
| `tests/test_waldo_service.py` | service 层单元测试 | 创建 |
| `frontend/app.py` | Streamlit 画廊页面 | 创建 |
| `frontend/requirements.txt` | 前端依赖（streamlit） | 创建 |
| `frontend/README.md` | 运行说明 | 创建 |

**关键实现约束（规格未尽事项，此处锁定）：**
- `original-images/` 里混有 `*_annotated.jpg`、`bbox/` 目录、`visualize.py`。`list_cases` 只收：是文件 + 后缀 ∈ {.jpg,.jpeg,.png} + 文件名（去后缀）**不以 `_annotated` 结尾**。
- 结果配对固定为 `outputs/{name}_result.jpg`（`visualize_node` 无论原图扩展名都存成 `.jpg`）。
- 目录用 `Path(__file__)` 相对项目根解析，不依赖运行时工作目录（Streamlit 工作目录可能不同）。
- `list_cases`/`get_case` 接受可选目录参数（带默认值），便于在 `tmp_path` 下测试。

---

### 任务 1：service 层（TDD）

**文件：**
- 创建：`service/__init__.py`
- 创建：`service/waldo_service.py`
- 测试：`tests/test_waldo_service.py`

- [ ] **步骤 1：创建空 package 标记文件**

创建 `service/__init__.py`，内容为空（仅让 `service` 成为可导入 package）。

- [ ] **步骤 2：编写失败的测试**

创建 `tests/test_waldo_service.py`：

```python
"""service.waldo_service 单元测试：纯函数，全部在 tmp_path 下构造目录。"""

from service.waldo_service import WaldoCase, list_cases, get_case


def _make_image(path):
    """写一个占位图片文件（service 只看扩展名/存在性，无需有效图像数据）。"""
    path.write_bytes(b"fake-image-bytes")


def test_list_cases_empty_when_images_dir_missing(tmp_path):
    missing = tmp_path / "nope"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    assert list_cases(images_dir=missing, outputs_dir=outputs) == []


def test_list_cases_pairs_results(tmp_path):
    images = tmp_path / "original-images"
    outputs = tmp_path / "outputs"
    images.mkdir()
    outputs.mkdir()
    _make_image(images / "1.jpg")
    _make_image(images / "2.jpg")
    _make_image(outputs / "1_result.jpg")  # 只有 1 有结果

    cases = list_cases(images_dir=images, outputs_dir=outputs)

    assert [c.name for c in cases] == ["1", "2"]
    case1 = cases[0]
    assert case1.has_result is True
    assert case1.result_path == str(outputs / "1_result.jpg")
    case2 = cases[1]
    assert case2.has_result is False
    assert case2.result_path is None


def test_list_cases_excludes_annotated_and_non_images(tmp_path):
    images = tmp_path / "original-images"
    outputs = tmp_path / "outputs"
    images.mkdir()
    outputs.mkdir()
    _make_image(images / "1.jpg")
    _make_image(images / "1_annotated.jpg")   # 必须排除
    (images / "visualize.py").write_text("# script")  # 非图片，排除
    (images / "bbox").mkdir()                  # 目录，排除

    cases = list_cases(images_dir=images, outputs_dir=outputs)

    assert [c.name for c in cases] == ["1"]


def test_list_cases_sorted_by_name(tmp_path):
    images = tmp_path / "original-images"
    outputs = tmp_path / "outputs"
    images.mkdir()
    outputs.mkdir()
    _make_image(images / "2.jpg")
    _make_image(images / "1.jpg")

    cases = list_cases(images_dir=images, outputs_dir=outputs)

    assert [c.name for c in cases] == ["1", "2"]


def test_get_case_hit_and_miss(tmp_path):
    images = tmp_path / "original-images"
    outputs = tmp_path / "outputs"
    images.mkdir()
    outputs.mkdir()
    _make_image(images / "1.jpg")

    hit = get_case("1", images_dir=images, outputs_dir=outputs)
    assert isinstance(hit, WaldoCase)
    assert hit.name == "1"

    miss = get_case("zzz", images_dir=images, outputs_dir=outputs)
    assert miss is None
```

- [ ] **步骤 3：运行测试验证失败**

运行：`python -m pytest tests/test_waldo_service.py -v`
预期：FAIL（`ModuleNotFoundError: No module named 'service.waldo_service'` 或导入错误）。

- [ ] **步骤 4：编写最少实现代码**

创建 `service/waldo_service.py`：

```python
"""WhereisWaldoAgent 共享服务层。

当前只提供"已有结果"的查询能力，供 Streamlit 前端使用；
将来 FastAPI 可直接复用 list_cases / get_case 包成 REST 接口。
"""

from dataclasses import dataclass
from pathlib import Path

# 相对本文件解析项目根：service/ 的上一级即项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "original-images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class WaldoCase:
    name: str                  # 原图文件名去后缀，如 "1"
    image_path: str            # 原图绝对路径
    result_path: str | None    # outputs/{name}_result.jpg；无则 None
    has_result: bool


def _is_source_image(p: Path) -> bool:
    """是否为可用作 case 的源图：文件 + 图片后缀 + 非 *_annotated。"""
    return (
        p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.stem.endswith("_annotated")
    )


def list_cases(
    images_dir: Path = IMAGES_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> list[WaldoCase]:
    """扫描 images_dir 下的源图，与 outputs_dir/{name}_result.jpg 配对。

    目录不存在时返回空列表，不抛异常。结果按 name 升序。
    """
    if not images_dir.exists():
        return []

    cases: list[WaldoCase] = []
    for p in sorted(images_dir.iterdir(), key=lambda x: x.name):
        if not _is_source_image(p):
            continue
        name = p.stem
        result = outputs_dir / f"{name}_result.jpg"
        has_result = result.is_file()
        cases.append(
            WaldoCase(
                name=name,
                image_path=str(p),
                result_path=str(result) if has_result else None,
                has_result=has_result,
            )
        )
    return cases


def get_case(
    name: str,
    images_dir: Path = IMAGES_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> WaldoCase | None:
    """按 name 返回单个 case；不存在返回 None。"""
    for case in list_cases(images_dir=images_dir, outputs_dir=outputs_dir):
        if case.name == name:
            return case
    return None
```

- [ ] **步骤 5：运行测试验证通过**

运行：`python -m pytest tests/test_waldo_service.py -v`
预期：5 个测试全部 PASS。

- [ ] **步骤 6：Commit**

```bash
git add service/__init__.py service/waldo_service.py tests/test_waldo_service.py
git commit -m "feat(service): add shared waldo result lookup layer"
```

---

### 任务 2：Streamlit 前端页面

**文件：**
- 创建：`frontend/requirements.txt`
- 创建：`frontend/app.py`
- 创建：`frontend/README.md`

- [ ] **步骤 1：安装 streamlit**

运行：`python -m pip install streamlit`
预期：安装成功（机器上当前未安装）。

- [ ] **步骤 2：写前端依赖文件**

创建 `frontend/requirements.txt`：

```
streamlit>=1.30
```

- [ ] **步骤 3：写 Streamlit 页面**

创建 `frontend/app.py`：

```python
"""WhereisWaldoAgent 最小前端：预览已有检测结果。

运行：streamlit run frontend/app.py
本页只读取 outputs/ 中已有的结果图；实时检测将在 FastAPI 版接入。
"""

import sys
from pathlib import Path

import streamlit as st

# 让位于项目根的 service/ 可被导入（Streamlit 工作目录不一定是项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.waldo_service import list_cases  # noqa: E402

st.set_page_config(page_title="Where's Waldo — 结果预览", layout="wide")
st.title("Where's Waldo Agent — 结果预览")
st.caption("本页只预览已有检测结果；实时检测将在 FastAPI 版接入。")

cases = list_cases()
if not cases:
    st.warning("没有找到任何图片（original-images/ 为空或不存在）。")
    st.stop()

labels = [f"{c.name}  {'✅ 有结果' if c.has_result else '— 未检测'}" for c in cases]
idx = st.sidebar.selectbox(
    "选择图片",
    range(len(cases)),
    format_func=lambda i: labels[i],
)
case = cases[idx]

col_src, col_res = st.columns(2)
with col_src:
    st.subheader("原图")
    st.image(case.image_path, use_container_width=True)
with col_res:
    st.subheader("检测结果")
    if case.has_result:
        st.image(case.result_path, use_container_width=True)
    else:
        st.info("尚未检测（outputs/ 中没有该图的结果）。")
```

- [ ] **步骤 4：写前端 README**

创建 `frontend/README.md`：

```markdown
# WhereisWaldoAgent 最小前端

基于 Streamlit 的结果预览页面。当前仅预览 `outputs/` 中已有的检测结果，
不实时调用 agent（实时检测将在将来的 FastAPI 版接入）。

## 运行

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

浏览器打开提示的本地地址（默认 http://localhost:8501）。
左侧选择图片，右侧查看红框标注结果；无结果的图会提示"尚未检测"。

## 架构

- `frontend/app.py` —— UI 层（将来换成 React）
- `service/waldo_service.py` —— 共享服务层（将来被 FastAPI 复用）

UI 只依赖 `service.waldo_service.list_cases()`，与检测逻辑解耦。
```

- [ ] **步骤 5：冒烟验证**

运行：`python -c "import ast; ast.parse(open('frontend/app.py', encoding='utf-8').read()); print('app.py OK')"`
预期：输出 `app.py OK`（语法正确）。

再手动运行 `streamlit run frontend/app.py`，浏览器确认：左侧能选图、右侧能显示原图，已有结果的图能显示红框结果图。（手动步骤，确认后关闭。）

- [ ] **步骤 6：Commit**

```bash
git add frontend/app.py frontend/requirements.txt frontend/README.md
git commit -m "feat(frontend): add minimal Streamlit result preview page"
```

---

## 自检结果

**1. 规格覆盖度：**
- 文件结构（service/ + frontend/ 平级）→ 任务 1、2 ✅
- service 契约（WaldoCase / list_cases / get_case）→ 任务 1 ✅
- Streamlit 画廊（选图 + 原图/结果并排 + 无结果提示）→ 任务 2 步骤 3 ✅
- 错误处理（目录缺失返回空、空列表友好提示）→ 任务 1 测试 + 任务 2 `st.stop()` ✅
- 测试策略（TDD service，手动冒烟 UI）→ 任务 1 TDD、任务 2 步骤 5 ✅
- YAGNI（不实时跑、不落盘 bbox、不引入 FastAPI/React）→ 计划未涉及，符合 ✅

**2. 占位符扫描：** 无 TODO / 待定 / "添加适当错误处理" 等占位；所有代码步骤含完整代码。✅

**3. 类型一致性：** `WaldoCase` 字段（name/image_path/result_path/has_result）、`list_cases`/`get_case` 签名在测试与实现、前端调用间一致。前端只用 `list_cases` 及 `WaldoCase.name/image_path/result_path/has_result`，均已定义。✅
