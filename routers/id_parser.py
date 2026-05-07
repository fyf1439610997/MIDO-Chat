import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from llm_config import build_llm_client
from prompt_templates import render_prompt

router = APIRouter(tags=["id-parse"])

GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

def _extract_docx_text(file_path: Path) -> str:
    doc = Document(str(file_path))
    lines: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_json_array(raw_content: str) -> list[dict[str, Any]]:
    content = raw_content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        raise ValueError("模型输出中未找到 JSON 数组。")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise ValueError("模型输出不是 JSON 数组。")
    return parsed


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip()
    if not stem:
        return "structured-id"
    safe = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", stem).strip("-_.")
    return safe or "structured-id"


def _normalize_roadmap(roadmap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _normalize_roadmap_with_timeline(roadmap, timeline_total_seconds=None)


def _parse_time_to_seconds(time_text: str) -> int | None:
    text = time_text.strip()
    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return max(0, minutes * 60 + seconds)
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return max(0, hours * 3600 + minutes * 60 + seconds)
    except ValueError:
        return None
    return None


def _format_seconds_to_mmss(total_seconds: int) -> str:
    safe = max(0, int(total_seconds))
    minutes = safe // 60
    seconds = safe % 60
    return f"{minutes:02d}:{seconds:02d}"


def _snap_floor_10s(value: int) -> int:
    return (value // 10) * 10


def _snap_ceil_10s(value: int) -> int:
    if value <= 0:
        return 10
    return ((value + 9) // 10) * 10


def _parse_range_to_10s(time_range: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)", time_range)
    if not match:
        return None
    start_sec = _parse_time_to_seconds(match.group(1))
    end_sec = _parse_time_to_seconds(match.group(2))
    if start_sec is None or end_sec is None:
        return None
    start_10s = _snap_floor_10s(start_sec)
    end_10s = _snap_ceil_10s(end_sec)
    if end_10s <= start_10s:
        end_10s = start_10s + 10
    return start_10s, end_10s


def _extract_timeline_total_seconds(transcript_timeline_json: str) -> int | None:
    if not transcript_timeline_json.strip():
        return None
    try:
        payload = json.loads(transcript_timeline_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    max_end = 0.0
    for item in payload:
        if not isinstance(item, dict):
            continue
        end_raw = item.get("end")
        try:
            end_val = float(end_raw)
        except (TypeError, ValueError):
            continue
        if end_val > max_end:
            max_end = end_val
    if max_end <= 0:
        return None
    return _snap_ceil_10s(int(max_end))


def _normalize_roadmap_with_timeline(
    roadmap: list[dict[str, Any]],
    timeline_total_seconds: int | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    count = max(1, len(roadmap))
    if timeline_total_seconds and timeline_total_seconds > 0:
        step = max(10, _snap_ceil_10s(timeline_total_seconds // count))
    else:
        step = 10

    for index, item in enumerate(roadmap, start=1):
        # 当有音频时间轴时，优先强制按时间轴范围做连续 10 秒分段，避免模型给出粗粒度区间。
        if timeline_total_seconds and timeline_total_seconds > 0:
            start_10s = (index - 1) * step
            end_10s = index * step
            if index == count:
                end_10s = timeline_total_seconds
            start_10s = _snap_floor_10s(start_10s)
            end_10s = _snap_ceil_10s(end_10s)
            if end_10s <= start_10s:
                end_10s = start_10s + 10
        else:
            parsed_range = _parse_range_to_10s(str(item.get("time", "")).strip())
            if parsed_range is not None:
                start_10s, end_10s = parsed_range
            else:
                start_10s = (index - 1) * step
                end_10s = index * step
        normalized.append(
            {
                "id": int(item.get("id", index)),
                "time": f"{_format_seconds_to_mmss(start_10s)}-{_format_seconds_to_mmss(end_10s)}",
                "phase": str(item.get("phase", "Unknown Phase")).strip(),
                "intent": str(item.get("intent", "Unknown Intent")).strip(),
            }
        )
    return normalized


def _mock_roadmap() -> list[dict[str, str | int]]:
    return [
        {"id": 1, "time": "00:00-05:00", "phase": "Introduction", "intent": "Concept Activation"},
        {"id": 2, "time": "05:00-15:00", "phase": "Group Work", "intent": "Collaboration"},
        {"id": 3, "time": "15:00-25:00", "phase": "Guided Practice", "intent": "Evidence-based Explanation"},
        {"id": 4, "time": "25:00-35:00", "phase": "Wrap-up", "intent": "Reflection and Consolidation"},
    ]


def _build_roadmap_from_llm(
    doc_text: str,
    transcript_text: str = "",
    subject_name: str = "生物",
    transcript_timeline_json: str = "",
) -> list[dict[str, Any]]:
    client, model, _provider = build_llm_client()
    if not client:
        return _mock_roadmap()

    system_prompt = render_prompt(
        "id_parse_system",
        subject_name=subject_name,
    )
    user_prompt = render_prompt(
        "id_parse_user",
        document_text=doc_text[:12000],
        transcript_text=(transcript_text[:12000] if transcript_text else "（暂无音频转写）"),
        transcript_timeline_json=(
            transcript_timeline_json[:12000] if transcript_timeline_json else "[]"
        ),
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = completion.choices[0].message.content if completion.choices else ""
    if not raw:
        raise ValueError("模型未返回内容。")
    parsed = _extract_json_array(raw)
    timeline_total_seconds = _extract_timeline_total_seconds(transcript_timeline_json)
    return _normalize_roadmap_with_timeline(parsed, timeline_total_seconds=timeline_total_seconds)


def _save_roadmap_json(roadmap: list[dict[str, Any]], source_filename: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = _safe_stem(source_filename)
    output_path = GENERATED_DIR / f"{base}-structured-id-{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(roadmap, f, ensure_ascii=False, indent=2)
    return output_path


def _friendly_parse_error_message(exc: Exception) -> str:
    raw = str(exc)
    lower = raw.lower()

    if "401" in lower or "incorrect api key" in lower or "invalid api key" in lower:
        return "模型服务鉴权失败：API Key 不正确或已失效。请联系管理员检查模型密钥配置。"
    if "403" in lower or "permission" in lower:
        return "模型服务权限不足：当前账号无权访问该模型。请联系管理员开通权限。"
    if "404" in lower and "model" in lower:
        return "模型名称配置错误：当前模型不可用。请联系管理员检查模型名称。"
    if "429" in lower or "rate limit" in lower or "quota" in lower:
        return "模型服务调用过于频繁或额度不足。请稍后重试，或联系管理员检查配额。"
    if "timeout" in lower or "connection" in lower or "network" in lower:
        return "连接模型服务超时或网络异常。请检查网络后重试。"

    return "教案解析暂时失败。请确认已上传有效 docx 文档，并稍后重试。"


@router.post("/api/parse_id")
async def parse_instructional_design(
    instructional_design_file: UploadFile = File(...),
    transcript_text: str = Form(default=""),
    subject_name: str = Form(default="生物"),
    transcript_timeline_json: str = Form(default=""),
) -> dict[str, Any]:
    filename = instructional_design_file.filename or ""
    if not filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="请上传 .docx 教学设计文件。")

    temp_path = GENERATED_DIR / f"upload-{datetime.now().strftime('%Y%m%d-%H%M%S')}.docx"
    content = await instructional_design_file.read()
    temp_path.write_bytes(content)

    try:
        doc_text = _extract_docx_text(temp_path)
        if not doc_text.strip():
            raise HTTPException(status_code=400, detail="文档为空，无法解析。")
        roadmap = _build_roadmap_from_llm(
            doc_text,
            transcript_text=transcript_text,
            subject_name=subject_name,
            transcript_timeline_json=transcript_timeline_json,
        )
        output_path = _save_roadmap_json(roadmap, filename)
        return {
            "roadmap": roadmap,
            "json_filename": output_path.name,
            "download_url": f"/api/parse_id/download/{output_path.name}",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_friendly_parse_error_message(exc)) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.get("/api/parse_id/download/{json_filename}")
async def download_structured_json(json_filename: str) -> FileResponse:
    safe_name = Path(json_filename).name
    target = GENERATED_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="找不到对应的结构化 JSON 文件。")
    return FileResponse(
        path=target,
        media_type="application/json",
        filename=safe_name,
    )


@router.get("/api/parse_id/files")
async def list_structured_json_files() -> dict[str, list[dict[str, str]]]:
    files = sorted(
        GENERATED_DIR.glob("*-structured-id-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "files": [
            {
                "filename": path.name,
                "download_url": f"/api/parse_id/download/{path.name}",
            }
            for path in files[:10]
        ]
    }
