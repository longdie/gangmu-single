# Gangmu Single

Gangmu Single 是 Gangmu v2 单文档流程的独立版本。它从一篇 LaTeX 或 Markdown 数学文献源码构建带来源的 MathIR 图谱，覆盖节点抽取、上下文保留、显式与隐式依赖、证明归属、拓扑检查和缓存复用。

输入入口读取源码目录。Markdown 中的 LaTeX 公式和原文位置会被保留，单文档流程不依赖跨文档 Embedding 或知识库发布。

## 安装

需要 Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python main.py init
```

在本地 `config.toml` 中填写模型地址、模型名和 API Key。该文件已被 Git 忽略，API Key 也可以通过 `YXAI_API_KEY` 环境变量提供。

## 一个最小例子

仓库包含 [examples/latex/main.tex](examples/latex/main.tex) 和 [examples/markdown/main.md](examples/markdown/main.md)。先离线固定源码：

```bash
python main.py prepare \
  --source examples/latex \
  --paper demo-latex \
  --version v1 \
  --main main.tex \
  --format latex
```

开始单文档抽取：

```bash
python main.py extract \
  --source examples/latex \
  --paper demo-latex \
  --version v1 \
  --main main.tex \
  --format latex \
  --session demo-latex-01 \
  --max-tokens 1000000
```

Markdown 的命令只需把源码目录和格式改成：

```bash
python main.py extract \
  --source examples/markdown \
  --paper demo-markdown \
  --version v1 \
  --main main.md \
  --format markdown \
  --session demo-markdown-01 \
  --max-tokens 1000000
```

结果默认写入 `results/<paper>/<production-fingerprint>/`，其中包含 `bundle/` 和 `run-report.json`。只检查已有缓存时使用 `--check-cache`。

查看图谱：

```bash
python main.py inspect results/demo-latex/<production-fingerprint>/bundle
```

## 代码结构

- `src/gangmu/single_document/` 保存单文档生产流程
- `src/gangmu/representation/` 保存 MathIR 模型、校验和图包存储
- `src/gangmu/runtime/` 保存来源固定、会话额度、请求缓存和模型调用
- `src/gangmu/latex_structure.py` 和 `src/gangmu/markdown_structure.py` 保存源码结构适配器
- `configs/` 和 `prompts/` 保存会影响生产指纹的策略文件

模型调用最多三个并发请求，技术重试和会话额度由运行时统一记录。源码快照、请求回执和结果缓存都写入本地忽略目录。
