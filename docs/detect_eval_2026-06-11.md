# detect 排序评测实验记录（2026-06-11）

## 背景

`main.py` 实跑 1.jpg / 2.jpg，pipeline 全程无报错但都未命中 Waldo。诊断发现 detect（`gpt-5.4-mini`）对几乎所有 patch 返回 `present=true, confidence≈0.99`（1.jpg 44/77、2.jpg 32/35），置信度排序退化为随机，真 Waldo 的 patch 排不进 verify 取用的 top-3，导致系统性 miss。

根因锁定在 `prompts.py` 的旧 `DETECT_PROMPT`：判据为「看到任意特征（红/白/条纹/帽子/眼镜）就 present=true」，在到处是红白色块的 Where's Waldo 图里几乎必中。

## 方法

新建评测脚本 `tests/eval_detect_ranking.py`，配套纯函数 `tests/patch_sampler.py`（含单测 `tests/test_patch_sampler.py`，10 项全绿）。

- **数据**：`original-images/N.jpg`（N=1..19，跳过无真值的 16）+ `original-images/bbox`（第 N 行 = `[x1,y1,x2,y2]` 真值）。
- **每图构造**：1 个正样本 patch（以真值框中心裁 `PATCH_PX` 方窗，含真 Waldo）+ 9 个负样本 patch（随机裁、与真值零重叠）。固定 `SEED=42`。
- **主指标 Top-3 命中率**：正样本置信度在该图 10 个 patch 中排进前 3 的图占比。**平局悲观处理**（正样本与负样本同分时排在并列者之后），使「全 0.99」破 prompt 被如实判为接近随机。
- **辅指标**：召回率（正样本 present=true 占比）、负样本误报率、正负样本平均置信度 gap。
- `PATCH_PX` 可命令行传参：`python tests/eval_detect_ranking.py 100`。

## 结果汇总（全部 gpt-5.4-mini，SEED=42，18 张评测图）

| # | prompt | patch | Top-3 命中(主) | 召回率 | 负样本误报 | gap |
|---|--------|:----:|:----:|:----:|:----:|:----:|
| 1 | 旧（任意特征） | 200px | 61.1% | **88.9%** | 57.4% | +0.311 |
| 2 | 判别式（红白横条纹同人） | 200px | 61.1% | 50.0% | 19.8% | +0.261 |
| 3 | 判别式 | 100px | 50.0% | 27.8% | 7.4% | +0.209 |
| 4 | 旧 | 200px | 55.6% | 88.9% | 56.8% | +0.276 |

> 注：run 4 与 run 1 同配置，仅 18.jpg 真值框换成修正中的 `[1200,200,1400,310]`（事后证明该框仍不准，见下）。

## 关键结论

1. **判别式 prompt 治标不治本（run 1→2）**：把判据收紧到「红白横条纹必须共址于同一个人躯干」后，负样本误报从 57%→20%（模型不再无脑说有），但**主指标 Top-3 纹丝未动（61.1%）**，且召回从 89% 崩到 50%。

2. **缩小 patch 反而更差（run 2→3）**：假设「100px 让 Waldo 相对变大、mini 更好认」被实测推翻——Top-3 跌到 50%、召回崩到 28%。排除裁切因素（2.jpg 真值 34×60 在 100px 内可完整容纳，正样本仍从 0.18 跌到 0.05）。原因是绝对分辨率/上下文损失：mini 需要周围场景才能确认「穿条纹衫的人」，只给 100px 躯干碎片认不出。**正面印证设计锚点「200×200 是 VLM 可靠识别 Waldo 的最小 patch 尺寸」**。

3. **瓶颈钉死**：`gpt-5.4-mini` 的判别力受 Waldo **绝对像素尺寸**（~30-50px、条纹 ~10-20px）限制，不是 prompt 措辞问题。**单靠 mini + prompt/patch 这条路已到顶。**

## pos 样本失败专项（本轮重点关注）

run 4（旧 prompt @200，修正中的 18 框）出现 2 个正样本失败（present=false，conf≈0.03）：

- **18.jpg pos_conf=0.03，rank 10/10**：框 `[1200,200,1400,310]`（200×110，远大于其它图的 ~30-90px）实际未框中 Waldo，模型正确判「无」。**证明该框仍是错的**——用户已据此再次修正为准确值（本文档不含再次重跑）。
- **4.jpg pos_conf=0.03**：真值 `[1484,279,1502,312]` = 18×33，是数据集中最小的 Waldo 之一，属真·难例（同一正样本在 run 1 曾得 0.88，见下方方差说明）。

## 方法学限制：temperature=1.0 致 run 间方差大

`gpt-5.4-mini` 当前以默认 `temperature=1.0` 调用，输出随机。同一正样本跨 run 置信度漂移显著（例：4.jpg 0.88↔0.03、13.jpg 0.98↔0.66、2.jpg HIT↔miss）。因此：

- run 之间 Top-3 的小幅波动（61.1%↔55.6%）很可能是采样噪声，而非配置差异——也印证「单张 18.jpg 标注误差对总体影响有限」。
- **后续评测应将 detect 的 temperature 调到 0（或最低）以保证可复现**，否则数字不可逐张严格比较。

## 剩余可选方向（100px 已排除）

1. **detect 换推理模型 `gpt-5.5` + 判别式 prompt**：在已验证够用的 200px 上用更强模型，最可能真正解决；代价是 ~17x 慢、贵（180 次调用约 30-50 分钟）。
2. **保留旧 prompt + 把 verify `TOP_K` 从 3 提到 8**：旧 prompt 召回 89%、miss 多排在第 4-8 名；提高 verify 候选数让真 Waldo 进入 verify，detect 不动、改一行；代价是 verify 调用翻倍。

## 产物清单

- `tests/patch_sampler.py` + `tests/test_patch_sampler.py`（纯函数 + 10 项单测）
- `tests/eval_detect_ranking.py`（评测 runner，支持命令行传 patch 边长）
- `original-images/bbox` 现有 18 张真值，可复用为后续整图 IoU 评测的基础
- 设计与判定细节见 `docs/superpowers/specs/2026-06-11-detect-discriminative-prompt-design.md`
- 判别式 `DETECT_PROMPT` 文本保存在上述设计文档中；因召回回归，未合入 `prompts.py`（main 仍为旧 prompt）

---

# 下午追加：交互式 prompt engineering（2026-06-11）

## 新增工具（quick 自查，给人用、可调 config）

- `tests/quick_detect_check.py` —— 召回自查：跑 `outputs/eval_patches/*_pos.jpg`，输出召回率 + 失败图号。
- `tests/quick_falsepos_check.py` —— 误检自查：跑 `*_neg*.jpg`，输出误检率 + 误检文件；支持按图号（`... 10`）只测单图。
- `tests/quick_config.py` —— 读 `config.json`，构造客户端，并提供 `run_repeats`（prompt engineering 时让模型多输出一行 **reason** 决策依据；正式 agent 不带 reason）。
- `config.json`（根目录）—— 集中可调参数：`provider/model/temperature/max_tokens/repeats/limit`。改 json 即可，无需动代码。

## 数据修正

- `original-images/bbox` 第 18 行多次修正，最终准确值 `[1250, 70, 1300, 120]`（50×50）。之前的错误标注一度造成 18 的"正样本失败"假象。

## prompt 演进与精确率/召回率前沿（全部实测）

| prompt | 模型 | 召回 | 误检(图10) | 备注 |
|--------|------|:---:|:---:|------|
| 轻：眼镜/帽子，任一特征即 true | mini | 100% | 100% | 特征不唯一 → 模型**脑补** → 误检爆表 |
| 中：必须红白条纹**衫** | mini | 33% | 0% | 衫在 200px 常看不清 → 召回崩 |
| 中：必须条纹衫 | gpt-5.5 | 67% | 11% | 有 3 张空响应假失败（token 截断）|
| 帽+眼镜**组合** | gpt-5.5 | 78% | 22% | 仍有 token 截断假失败 |
| **不列特征 + "可能模糊"提示** | **gpt-5.5** | **88.9%** | ~20% | **最佳；token 修复后无假失败** |

## 关键结论

1. **不列具体特征反而最好**：模型自身就认识 Waldo，枚举特征只帮倒忙——列帽/眼镜诱导脑补（误检爆），列条纹衫逼其过严（召回崩）。一句"Waldo 可能小/被遮挡/模糊，仔细看"的整体判断，召回最高。
2. **领域真相（用户提供）**：本数据集里 **红白条纹帽 + 眼镜总可见，条纹衫只偶尔出现且模糊** ——所以"以条纹衫为闸门"方向是错的。
3. **confidence 语义已修**：明确为"Waldo 存在的概率"且必须与 present 一致（修掉了 present=false / conf=0.98 的矛盾）。
4. **gpt-5.5 token 截断 bug 已修**：推理 token 吃光默认 1024 预算 → 返回空响应被判 present=false/conf=0。修法：`config.json` 加 `max_tokens=4096`、`build_vlm` 透传。修后 2/13 等假失败消失。
5. **temperature**：mini 设 0 求可复现；gpt-5.5 是推理模型必须 1。

## 当前工作区状态（未 commit）

- `prompts.py` —— `DETECT_PROMPT` = feature-free 版（不列特征 + 模糊提示），confidence 语义已修。**未 commit**。
- `config.json` —— 当前 `gpt-5.5 / temp=1 / max_tokens=4096 / limit=5`。
- `tests/quick_*.py` + `tests/quick_config.py` —— 新增，未 commit。
- `original-images/bbox`、`18_annotated.jpg` —— 数据修正，未 commit。
- 正式 `agent/nodes/detect.py` 仍为 `gpt-5.4-mini`，**未改**。

---

# 明日工作（2026-06-12）

1. **存档 commit**：把今天有效成果落库——feature-free `DETECT_PROMPT`、`max_tokens` 修复、quick 工具四件套、`config.json`、bbox 修正。建议拆成几个语义化 commit（英文）。
2. **mini 上测 feature-free 版**（便宜）：把 `config.json` 切回 `gpt-5.4-mini / temp=0`，跑 recall + falsepos。若 mini 在 feature-free 下召回也够用，就不必上 gpt-5.5，省掉 17x 成本。**优先做**。
3. **更可信的误检率**：目前只测了图10抽5张，FP≈20% 样本太小。跨多图多抽负样本（`config.json` 的 `limit` + 跑全 `quick_falsepos_check.py`），拿到稳定 FP。
4. **落地决策**：是否把 `agent/nodes/detect.py` 的 `VLM_MODEL` 换 gpt-5.5（权衡 ~17x 慢/贵）；若换，记得 detect 节点也要把 max_tokens 调高（同款 token 截断坑）。
5. **真失败兜底**：10、11.jpg 是 feature-free + gpt-5.5 下仅剩的真漏检；可结合 verify 阶段或放大送检（见 `outputs/inspect/` 思路）专门处理。
6. **整图验证**：上述都在 patch 级；最终应跑一遍完整 `main.py` 看端到端能否命中（detect 改动 + verify 联动）。

