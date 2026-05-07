import json
from collections.abc import AsyncIterator, Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_config import build_llm_client
from prompt_templates import render_prompt

router = APIRouter(tags=["optimize"])


class OptimizePayload(BaseModel):
    mode: str = Field(default="chat")
    subject_name: str = Field(default="生物", min_length=1)
    teaching_intent: str = Field(..., min_length=1)
    selected_time_range: str = Field(..., min_length=1)
    timeline_features: str = Field(..., min_length=1)
    class_memory_context: str = Field(default="（未提供班级历史基线记忆）")
    class_memory_filename: str = Field(default="")
    user_prompt: str = Field(default="")
    metrics: dict[str, object]


def _format_metrics(metrics: dict[str, object]) -> str:
    return json.dumps(metrics, ensure_ascii=False, indent=2)


def _mock_strategy_stream(payload: OptimizePayload) -> Iterator[str]:
    if payload.mode == "chat":
        fallback_text = (
            f"当前学科：{payload.subject_name}\n"
            f"你问的是：{payload.user_prompt or '（未提供问题）'}\n"
            "这是一般问答模式，我会优先给出简洁数据解读。"
        )
        for ch in fallback_text:
            yield f"data: {json.dumps({'delta': ch}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    fallback_text = (
        "## 断层诊断 (Gap Diagnosis)\n"
        f"- 当前学科：{payload.subject_name}\n"
        f"- 当前教学环节意图：{payload.teaching_intent}\n"
        f"- 选定时段：{payload.selected_time_range}\n"
        f"- 关键分布特征：{payload.timeline_features}\n"
        "- 行为参与虽存在活跃占比，但被动与扰动成分仍需压降。\n"
        "- ICAP 中 Off-task 抬升提示任务目标与讨论行为发生偏移。\n\n"
        "## 语境校准 (Contextual Calibration)\n"
        "1. 在该环节开始前发放 1 页任务锚点单，明确证据标准与产出模板。\n"
        "2. 每 3 分钟执行一次小组快检：结论-证据-反例是否齐备。\n"
        "3. 将 Off-task 风险高的小组改为双角色协作（解释者 + 质询者）。\n"
        "4. 结束前 1 分钟进行全班口头回收，核对是否达成环节意图。\n"
    )
    for ch in fallback_text:
        yield f"data: {json.dumps({'delta': ch}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _openai_strategy_stream(payload: OptimizePayload) -> Iterator[str]:
    client, model, _provider = build_llm_client()
    if not client:
        yield from _mock_strategy_stream(payload)
        return

    mode = payload.mode if payload.mode in {"chat", "optimize"} else "chat"
    if mode == "optimize":
        system_prompt = render_prompt("optimize_system")
        user_prompt = render_prompt(
            "optimize_user",
            subject_name=payload.subject_name,
            teaching_intent=payload.teaching_intent,
            selected_time_range=payload.selected_time_range,
            timeline_features=payload.timeline_features,
            class_memory_context=payload.class_memory_context or "（未提供班级历史基线记忆）",
            class_memory_filename=payload.class_memory_filename or "（未提供）",
            metrics_json=_format_metrics(payload.metrics),
            user_prompt=payload.user_prompt or "（无）",
        )
    else:
        system_prompt = render_prompt("chat_system")
        user_prompt = render_prompt(
            "chat_user",
            subject_name=payload.subject_name,
            teaching_intent=payload.teaching_intent,
            selected_time_range=payload.selected_time_range,
            timeline_features=payload.timeline_features,
            class_memory_context=payload.class_memory_context or "（未提供班级历史基线记忆）",
            class_memory_filename=payload.class_memory_filename or "（未提供）",
            metrics_json=_format_metrics(payload.metrics),
            user_prompt=payload.user_prompt or "请对当前数据做简要解读。",
        )

    stream = client.chat.completions.create(
        model=model,
        stream=True,
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


async def _event_generator(payload: OptimizePayload) -> AsyncIterator[str]:
    try:
        for event in _openai_strategy_stream(payload):
            yield event
    except Exception as exc:  # pragma: no cover - runtime fallback
        message = f"\n\n> 生成失败：{exc}"
        yield f"data: {json.dumps({'delta': message}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/api/optimize")
async def optimize(payload: OptimizePayload) -> StreamingResponse:
    return StreamingResponse(_event_generator(payload), media_type="text/event-stream")
