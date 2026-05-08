# MIDO-Chat

[English](./README_EN.md) | 中文

MIDO-Chat（Multimodal Instructional Design Optimization）是一个面向课堂场景的多模态教学优化工作台。  
系统将视频检测（行为/情感）、音频转写与 ICAP、教学设计结构化和 LLM 对话整合到同一界面，用于生成可执行的教学改进建议。

## 功能概览

- 一键 `Data Processing` 流程：视频上传、目标检测、音频转写、ICAP、教案结构化
- 三维时序图：Behavioral / Emotional / Cognitive(ICAP) 按 30 秒粒度展示
- 支持多数据源切换：Video Analysis Results / Local Timeline JSON / Mock Data
- 教学诊断与对话：基于选定时间段 + 教学意图 + 指标数据进行流式输出（SSE）
- 班级记忆文件：支持加载本地“班级历史基线记忆”以增强个性化建议
- 过程文件下载：音频、转写、投入度 CSV/JSON、结构化教学设计 JSON

## 项目展示图

![MIDO-Chat UI Overview](./docs/images/mido-chat-overview.png)

> 请将展示图放到 `docs/images/mido-chat-overview.png`。  
> 如需多图展示，可继续在该目录下新增图片并在 README 中追加 Markdown 图片链接。

## 技术栈

- 后端：Python, FastAPI, Pydantic, OpenAI SDK, Jinja2, Uvicorn
- 前端：HTML, Vanilla JavaScript, Tailwind CSS, Chart.js
- 多媒体：FFmpeg（音频提取/分段）
- 视觉分析：RT-DETR-DHSA（项目内子目录调用）

## 项目结构

```text
MIDO-Chat/
├─ main.py
├─ llm_config.py
├─ llm.private.example.json
├─ llm.private.json          # 本地私密配置（已忽略）
├─ prompt_templates.py
├─ requirements.txt
├─ routers/
│  ├─ id_parser.py
│  ├─ metrics.py
│  ├─ optimize.py
│  └─ video.py
├─ templates/
│  └─ index.html
├─ generated/                # 运行产物（已忽略）
└─ RT-DETR-DHSA/
```

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置模型密钥

复制示例文件并填写你自己的密钥：

```bash
cp llm.private.example.json llm.private.json
```

`llm.private.json` 已在 `.gitignore` 中，不会被提交。  
也可使用环境变量作为兜底。

### 3) 准备 FFmpeg（必需）

请确保命令行可直接执行：

```bash
ffmpeg -version
ffprobe -version
```

### 4) 启动服务

```bash
uvicorn main:app --reload
```

访问：

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 核心 API

### 视频处理与进度

- `GET /api/video/limits`
- `POST /api/video/upload/init`
- `POST /api/video/upload/chunk`
- `POST /api/video/upload/complete`
- `POST /api/video/extract_audio`
- `POST /api/video/transcribe_audio`
- `GET /api/video/transcribe_progress/{upload_id}`
- `GET /api/video/vision_progress/{upload_id}`

### 结果下载

- `GET /api/video/audio/download/{audio_filename}`
- `GET /api/video/transcript/download/{transcript_filename}`
- `GET /api/video/vision_metrics/download/{upload_id}/csv`
- `GET /api/video/vision_metrics/download/{upload_id}/json`
- `GET /api/parse_id/download/{json_filename}`

### 教学设计与策略

- `POST /api/parse_id`
- `POST /api/metrics_timeline/video`
- `POST /api/optimize`（SSE 流式）

## 班级历史基线记忆文件

界面支持“Load Class Memory File”按钮，推荐 JSON 格式。  
模板文件位于：

- `generated/class_memory_template.json`

该文件会作为上下文传给 `/api/optimize`，用于让建议更贴合具体班级画像。

## 提示词与可配置项

所有提示词集中在：

- `prompt_templates.py`

主要模板：

- `id_parse_system` / `id_parse_user`
- `optimize_system` / `optimize_user`
- `chat_system` / `chat_user`
- `icap_window_system` / `icap_window_user`

## 安全与忽略策略

默认已忽略：

- `llm.private.json`
- `generated/`
- `__pycache__/`
- 测试案例目录模式（`*测试*/`, `*案例*/`, `test_cases/` 等）

## 常见问题

- **策略输出看起来像 mock？**  
  请检查密钥配置；无可用密钥时会走回退逻辑。

- **视频流程卡住或失败？**  
  优先检查 FFmpeg 是否可用、GPU/模型路径是否正常、以及上传文件是否完整。

- **为什么有些时间段按钮灰掉不可点击？**  
  当前数据源在该时间段没有可用窗口数据。

## 许可

当前仓库未附带开源许可证。如需开源，请补充 `LICENSE`。
