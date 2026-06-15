"""WhereisWaldoAgent 最小前端（架构占位版）：预览已有检测结果。

运行：streamlit run frontend/app.py

定位：UI 层。只依赖 service.waldo_service 的契约，不碰检测逻辑。
将来此目录会被 React 替换，service 层保持不变。
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
