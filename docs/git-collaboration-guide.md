# 多人协作 Git 工作流指南

> 场景：本仓库 `Where-is-Waldo-personal` 原本由 **jinchi（维护者）** 单人开发，所有提交直接打在 `main` 上。现在 **同学（新协作者，已是 collaborator）** 加入。本指南带你们走一遍从「同步基线」到「日常协作」的完整流程。

---

## 0. 当前现状（开始前先理解）

- 维护者本地 `main` **领先 GitHub 上的 `origin/main` 27 个 commit**（近期所有工作都还没推上去）。
- 同学是从 GitHub clone 的旧 `main`，并基于它建了个**测试分支（无实质改动）**。
- 结论：**GitHub 上的版本是旧的**。必须先让维护者把本地 `main` 推上去，GitHub 才是最新基线，同学再以它为基础工作。

角色约定：

| 角色 | 是谁 | 职责 |
|------|------|------|
| 维护者 | jinchi | 先建立最新基线；review 合并 PR |
| 协作者 | 同学 | 基于最新 `main` 开 feature 分支干活，提 PR |

---

## 阶段 1：维护者建立干净的最新基线（jinchi 执行）

### 1.1 整理工作区改动

本地有几类未提交改动，分别处理：

```bash
# (a) 反跟踪「生成类」标注图——.gitignore 已声明忽略，本不该入库（可由 visualize.py 重新生成）
git rm --cached original-images/*_annotated.jpg

# (b) 暂存真值校正 + 新脚本 + 已确认的数据集删除 + .gitignore 更新
git add .gitignore
git add original-images/bbox
git add scripts/gemini_limit.py scripts/gemini_box_debug.py
git add -u images_quicktests/          # 记录这些 png 的删除（已确认是有意清理）
```

> `.claude/`（本地 assistant 配置/技能）已加入 `.gitignore`，不会被推送——同学不使用，无需共享。

### 1.2 分成几个清晰的 commit

```bash
git commit -m "chore: ignore .claude local tooling and untrack generated annotated images"

git add original-images/bbox
git commit -m "fix(data): correct 19.jpg Waldo ground-truth bbox"

git add scripts/gemini_limit.py scripts/gemini_box_debug.py
git commit -m "feat(scripts): add Gemini patch-size limit and bbox-debug probes"

git add -u images_quicktests/
git commit -m "chore(data): remove unused images_quicktests png set"
```

> 提交信息全英文（项目约定）。一个 commit 只做一件事，方便 review 和回溯。

### 1.3 推送到 GitHub，建立最新基线

```bash
git push origin main
```

推完后 GitHub 的 `main` 就和你电脑一致了。这是整个协作的起点。

---

## 阶段 2：同学同步到最新基线（同学执行）

同学的测试分支是空的，直接丢弃重来即可：

```bash
git fetch origin                       # 拉取最新远程信息
git checkout main
git reset --hard origin/main           # 把本地 main 强制对齐到最新（仅当本地 main 无想保留的改动时用）
git branch -D <他的测试分支名>          # 删掉那个测试分支
```

> ⚠️ `reset --hard` 会丢弃本地未提交改动。同学的 main 没有实质改动，所以安全；养成习惯：只在确认本地无要保留内容时用。

现在双方的 `main` 都和 GitHub 一致，可以开始正常协作。

---

## 阶段 3：日常协作——feature 分支 + PR（核心，双方都这样做）

**铁律：谁都不再直接往 `main` 上 commit/push。** 每个人在自己的 feature 分支干活，通过 PR 合并。

### 3.1 开一个新分支干活

```bash
git checkout main
git pull origin main                   # 先确保从最新 main 出发
git checkout -b feature/你的功能名       # 例：feature/detect-bbox  或  feature/segment-refactor
```

分支命名建议：`feature/xxx`（新功能）、`fix/xxx`（修 bug）、`docs/xxx`（文档）。

### 3.2 干活 → 提交 → 推分支

```bash
# ... 改代码 ...
git add <改动的文件>
git commit -m "feat(detect): return bbox when Waldo present"
git push -u origin feature/你的功能名    # 第一次推带 -u，之后直接 git push
```

### 3.3 在 GitHub 上开 PR

- 打开仓库页面，会提示 “Compare & pull request”，点它。
- base = `main`，compare = 你的 feature 分支。
- 写清楚标题和说明，指定对方 review。
- review 通过后点 **Merge**，合并进 `main`。
- 合并后删掉该 feature 分支（GitHub 上点 Delete branch；本地 `git branch -d feature/你的功能名`）。

---

## 阶段 4：保持分支最新 / 处理「main 动了」

当对方的 PR 先合并进 `main`，你的 feature 分支就落后了。合并前先把你的分支更新到最新 `main`：

```bash
git checkout feature/你的功能名
git fetch origin
git rebase origin/main                 # 把你的提交「搬」到最新 main 之上
# 若有冲突：手动改文件 → git add <文件> → git rebase --continue
git push --force-with-lease            # rebase 改写了历史，需强推自己的分支（--force-with-lease 比 --force 安全）
```

> `--force-with-lease` 只在远程分支没被别人动过时才强推，避免覆盖他人提交。**只对自己的 feature 分支用，永远不要对 `main` 强推。**

---

## 常用命令速查

| 目的 | 命令 |
|------|------|
| 看当前分支 / 状态 | `git status -sb` |
| 看所有分支 | `git branch -a` |
| 看本地与远程差多少 | `git status -sb`（首行 ahead/behind） |
| 拉最新 main | `git checkout main && git pull origin main` |
| 开新分支 | `git checkout -b feature/xxx` |
| 推新分支 | `git push -u origin feature/xxx` |
| 更新分支到最新 main | `git fetch && git rebase origin/main` |
| 撤销工作区改动（未 add） | `git restore <文件>` |

## 几条协作约定

1. `main` 永远保持可运行；只通过 PR 合并，不直接 push。
2. commit 信息全英文，一个 commit 一件事。
3. 开工前先 `git pull` / `git fetch`，别基于旧 main 干活。
4. 不要提交 `.env`、`outputs/`、`.claude/` 等本地/生成文件（已在 `.gitignore`）。
5. 冲突不可怕：rebase 时逐个文件解决，`git add` 后 `git rebase --continue`。
</content>
