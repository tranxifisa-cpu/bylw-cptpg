# 中文论文与 Overleaf 同步

GitHub 的 `main` 保存论文源文件，现有 Overleaf 项目用于协作和编译。

- 项目：<https://www.overleaf.com/project/6946a3f58266e8759a116aa4>
- 目录对应：`output_paper/应统作文模板/` → Overleaf 的 `中文初稿v1/`
- 独立克隆默认位置：仓库旁的 `paper-overleaf`，本机为 `D:\project\paper-overleaf`

## 首次连接

在自己的 PowerShell 终端进入 `D:\project\bylw`，运行：

```powershell
python scripts/sync_overleaf.py setup
```

输入新建的 Overleaf Git token，只粘贴一次后按 Enter，输入过程不显示。凭据交给 Git Credential Manager 保存，不进入网址、源码或命令历史。Windows 使用现有的 `manager` 凭据助手；克隆时使用 OpenSSL，以兼容本机的 TLS 连接。同步时自动读取项目的默认分支。

首次连接会克隆原项目、核对论文内容并记录同步基线。已有文件不同就停止，确认两边的改动后再建立基线。克隆的提交钩子会阻止 Overleaf token 进入 Git 历史。

## 更新论文

先将云端修改提交、合入 GitHub 的 `main`，再运行：

```powershell
python scripts/sync_overleaf.py status
python scripts/sync_overleaf.py push
```

`status` 拉取两端并列出论文文件差异；`push` 拉取 GitHub 的 `main`，提交并推送中文论文目录。读取的是 Git 中已提交的版本，不会上传工作区里尚未提交的内容。现有 Overleaf 项目内的其他目录保持原样。

若导师先在 Overleaf 修改了中文稿，脚本会指出哪些文件偏离上次同步基线。先把这些改动合入 GitHub，再执行推送；不要同时在两端修改同一稿件。

同步包括论文的 TeX、参考文献、模板和图表等已跟踪文件；忽略 LaTeX 编译辅助文件及未用于编译的声明 Word 源文件。既有的论文 PDF 和图表 PDF 会随源目录同步。

在另一台电脑或云端执行时，先运行一次 `setup`，再使用相同命令。可用 `--overleaf-dir` 指定独立克隆位置。不要把凭据或独立克隆提交到研究仓库。
