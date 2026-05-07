# MIDO-Chat

MIDO-Chat（Multimodal Instructional Design Optimization）是一个基于 FastAPI + 原生前端的教学设计优化工作台。  
系统通过“三维参与度分布（Behavioral / Emotional / Cognitive-ICAP）”与“当前教学意图”结合，生成具有可操作性的课堂优化策略。

## 核心能力

- 三维群体参与度可视化：左侧面板以堆叠进度条展示全班百分比分布
- Mock 数据引擎：`/api/metrics` 每次返回结构化比例数据（各维度和为 1.0）
- 动态刷新：前端每 3 秒自动拉取最新指标并平滑更新 UI
- 策略生成：`/api/optimize` 接收教学意图 + 最新指标，调用大模型生成策略
- SSE 流式输出：后端以 `StreamingResponse` 持续返回，前端实时打字机渲染

## 技术栈

- 后端：Python 3.10+、FastAPI、Pydantic、OpenAI SDK、Uvicorn、Jinja2
- 前端：HTML5、原生 JavaScript（ES6+）、Tailwind CSS（CDN）

## 项目结构

```text
MIDO-Chat/
├─ main.py
├─ llm_config.py
├─ llm.private.example.json
├─ requirements.txt
├─ README.md
├─ routers/
│  ├─ __init__.py
│  ├─ id_parser.py
│  ├─ metrics.py
│  └─ optimize.py
└─ templates/
   └─ index.html
```

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置私密模型文件（推荐）

项目根目录有两个文件：

- `llm.private.example.json`：多模型示例
- `llm.private.json`：你的私密配置（已加入 `.gitignore`，不会提交）

推荐将示例复制到私密文件后填写：

```json
{
  "active_profile": "openai",
  "profiles": {
    "openai": {
      "api_key": "你的-openai-key",
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4o-mini"
    },
    "deepseek": {
      "api_key": "你的-deepseek-key",
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat"
    },
    "qwen": {
      "api_key": "你的-qwen-key",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "model": "qwen-plus"
    }
  }
}
```

切换模型只需改一处：

- `active_profile` 改成 `openai` / `deepseek` / `qwen`

### 3) 环境变量（可选兜底）

在 Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

> 未设置 `OPENAI_API_KEY` 时，`/api/optimize` 会自动回退为 mock 流式输出，方便本地演示。

> 当 `llm.private.json` 存在时，优先读取该文件；环境变量作为兜底。

### 4) 启动服务

```bash
uvicorn main:app --reload
```

启动后访问：

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API 说明

### 视频大文件与音频分离

为支持 1-10GB 视频，后端采用分片上传方案（默认上限 10GB）：

- `GET /api/video/limits`：获取上传上限
- `POST /api/video/upload/init`：初始化上传任务
- `POST /api/video/upload/chunk`：上传分片
- `POST /api/video/upload/complete`：合并分片
- `POST /api/video/extract_audio`：从视频分离音频（mp3/wav）
- `POST /api/video/transcribe_audio`：使用小模型进行音频转写（含时间片段）
- `GET /api/video/audio/download/{audio_filename}`：下载音频
- `GET /api/video/transcript/download/{transcript_filename}`：下载转写 JSON

> 音频分离依赖 `ffmpeg`，请先在服务器安装并确保命令行可用。
> 音频转写模型可在 `llm.private.json` 中配置：
> - `asr_profile`：用于转写的模型供应商档位（建议 `gpt4o`）
> - `asr_model`：转写模型名（默认 `gpt-4o-mini-transcribe`）
> - 超长音频会自动触发分块转写并合并结果

### `GET /api/metrics`

返回三维参与度分布（Mock）：

```json
{
  "behavioral": {
    "Active": 0.52,
    "Passive": 0.36,
    "Disruptive": 0.12
  },
  "emotional": {
    "Positive": 0.46,
    "Neutral": 0.41,
    "Frustrated": 0.13
  },
  "icap": {
    "Interactive": 0.18,
    "Constructive": 0.29,
    "Active": 0.23,
    "Passive": 0.20,
    "Off-task": 0.10
  }
}
```

### `POST /api/optimize`

请求体：

```json
{
  "teaching_intent": "通过小组合作完成细胞结构的证据化比较与解释",
  "metrics": {
    "behavioral": {"Active": 0.5, "Passive": 0.4, "Disruptive": 0.1},
    "emotional": {"Positive": 0.4, "Neutral": 0.5, "Frustrated": 0.1},
    "icap": {"Interactive": 0.2, "Constructive": 0.3, "Active": 0.2, "Passive": 0.2, "Off-task": 0.1}
  }
}
```

响应类型：

- `text/event-stream`（SSE）
- 数据帧格式：`data: {"delta":"..."}`，结束标记：`data: [DONE]`

## 前端交互逻辑

- 页面加载后立即请求一次 `/api/metrics`
- 每 3 秒定时拉取最新指标并更新堆叠进度条与百分比
- 点击“生成优化策略”后，POST 到 `/api/optimize`
- 逐块解析 SSE 数据并实时追加到右侧策略区，自动滚动到底部

## 模型提示词设计（后端）

项目所有提示词统一放在 `prompt_templates.py`，并通过变量模板渲染：

- `id_parse_system` / `id_parse_user`：用于教案结构化
- `optimize_system` / `optimize_user`：用于诊断与策略生成

你可以在模板里用变量占位，例如 `{assistant_name}`、`{document_text}`、`{teaching_intent}`。

`/api/optimize` 的输出逻辑要求模型按两步分析：

1. **断层诊断（Gap Diagnosis）**：对比教学意图与实际分布，识别异常占比与风险
2. **语境校准（Contextual Calibration）**：给出具体、可执行、可落地的教案改造动作

调用位置：

- `routers/id_parser.py`：通过 `render_prompt("id_parse_*", ...)` 注入变量
- `routers/optimize.py`：通过 `render_prompt("optimize_*", ...)` 注入变量

## 常见问题

- **Q: 为什么策略输出不是模型真实结果？**  
  A: 请确认 `OPENAI_API_KEY` 已配置；未配置时会进入 mock 流式回退。

- **Q: SSE 没有实时显示？**  
  A: 检查浏览器控制台是否有网络错误，确认后端返回 `text/event-stream`，并且前端在读取 `ReadableStream`。

## 后续迭代建议

- 将 `metrics.py` 的 Mock 改为随机扰动/回放数据引擎
- 将右侧策略区升级为 Markdown 渲染（标题、列表、强调）
- 增加策略生成过程状态（排队中、生成中、完成、失败）
- 引入鉴权与会话管理，支持多班级、多课程并行分析

## 许可

当前仓库未声明开源许可，如需开源请补充 `LICENSE` 文件。
