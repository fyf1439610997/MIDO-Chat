import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(tags=["metrics"])
WORK_DIR = Path("generated/video")
UPLOAD_DIR = WORK_DIR / "uploads"
VISION_METRICS_DIR = WORK_DIR / "vision_metrics"


class VideoMetricsRequest(BaseModel):
    upload_id: str = Field(..., min_length=1, description="视频上传任务ID")
    time_range: str = Field(..., description="时间区间，格式如 00:00-05:00")
    subject_name: str = Field(default="生物")


def _to_seconds(mmss: str) -> int:
    minute, second = mmss.split(":")
    return int(minute) * 60 + int(second)


def _to_mmss(value: int) -> str:
    minute = value // 60
    second = value % 60
    return f"{minute:02d}:{second:02d}"


def _aligned_30s_points(start_sec: int, end_sec: int, step: int = 30) -> list[int]:
    """Return globally aligned 30s timeline points within (start, end]."""
    if end_sec <= start_sec:
        return []
    first = ((start_sec + step - 1) // step) * step
    if first <= start_sec:
        first += step
    return list(range(first, end_sec + 1, step))


def _read_upload_meta(upload_id: str) -> dict | None:
    legacy_path = UPLOAD_DIR / upload_id / "meta.json"
    if legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    for folder in UPLOAD_DIR.iterdir():
        if not folder.is_dir():
            continue
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if str(meta.get("upload_id", "")) == upload_id:
            return meta
    return None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _build_timeline_from_vision(upload_id: str, time_range: str) -> dict[str, object]:
    meta = _read_upload_meta(upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="上传任务不存在，请重新上传视频。")

    vision_status = str(meta.get("vision_status", "idle"))
    json_path_raw = str(meta.get("vision_json_path", "")).strip()
    folder_name = str(meta.get("upload_folder", upload_id))
    json_path = Path(json_path_raw) if json_path_raw else (VISION_METRICS_DIR / folder_name / "classroom_30s_stats.json")

    if vision_status != "completed" or not json_path.exists():
        detail = str(meta.get("vision_error", "")).strip()
        if vision_status == "failed":
            raise HTTPException(status_code=500, detail=f"视觉分析失败：{detail or '未知错误'}")
        raise HTTPException(status_code=409, detail="视觉分析尚未完成，请稍后重试。")

    with json_path.open("r", encoding="utf-8") as f:
        windows = json.load(f)
    if not isinstance(windows, list):
        raise HTTPException(status_code=500, detail="视觉分析结果格式无效。")

    try:
        start_text, end_text = [part.strip() for part in time_range.split("-")]
        start_sec = _to_seconds(start_text)
        end_sec = _to_seconds(end_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"时间区间格式错误：{time_range}") from exc
    if end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="时间区间无效：结束时间必须大于开始时间。")

    aligned_points = _aligned_30s_points(start_sec, end_sec, step=30)
    if not aligned_points:
        raise HTTPException(status_code=404, detail="所选时间段无可用视觉统计数据。")

    by_end_second: dict[int, dict] = {}
    for item in windows:
        if not isinstance(item, dict):
            continue
        end_second = int(round(float(item.get("end_second", 0.0) or 0.0)))
        by_end_second[end_second] = item

    matched: list[dict] = []
    for sec in aligned_points:
        item = by_end_second.get(sec)
        if item is not None:
            matched.append(item)
    if not matched:
        raise HTTPException(status_code=404, detail="所选时间段无可用视觉统计数据。")

    labels: list[str] = []
    behavioral_active: list[float] = []
    behavioral_passive: list[float] = []
    behavioral_disruptive: list[float] = []
    emotional_positive: list[float] = []
    emotional_neutral: list[float] = []
    emotional_frustrated: list[float] = []
    icap_interactive: list[float] = []
    icap_constructive: list[float] = []
    icap_active: list[float] = []
    icap_passive: list[float] = []
    icap_off_task: list[float] = []

    for item in matched:
        end_second = int(round(float(item.get("end_second", 0.0) or 0.0)))
        labels.append(_to_mmss(end_second))

        behavior_pct = item.get("behavior_percentages") or {}
        expression_pct = item.get("expression_percentages") or {}
        icap_pct = item.get("icap_percentages") or {}

        b_active = _clamp01(float(behavior_pct.get("High", 0.0)) / 100.0)
        b_passive = _clamp01(float(behavior_pct.get("Medium", 0.0)) / 100.0)
        b_disruptive = _clamp01(float(behavior_pct.get("Low", 0.0)) / 100.0)
        b_sum = b_active + b_passive + b_disruptive
        if b_sum > 0:
            b_active, b_passive, b_disruptive = b_active / b_sum, b_passive / b_sum, b_disruptive / b_sum

        e_positive = _clamp01(float(expression_pct.get("Positive", 0.0)) / 100.0)
        e_neutral = _clamp01(float(expression_pct.get("Neutral", 0.0)) / 100.0)
        e_frustrated = _clamp01(float(expression_pct.get("Negative", 0.0)) / 100.0)
        e_sum = e_positive + e_neutral + e_frustrated
        if e_sum > 0:
            e_positive, e_neutral, e_frustrated = e_positive / e_sum, e_neutral / e_sum, e_frustrated / e_sum

        if icap_pct:
            interactive = _clamp01(float(icap_pct.get("Interactive", 0.0)) / 100.0)
            constructive = _clamp01(float(icap_pct.get("Constructive", 0.0)) / 100.0)
            active = _clamp01(float(icap_pct.get("Active", 0.0)) / 100.0)
            passive = _clamp01(float(icap_pct.get("Passive", 0.0)) / 100.0)
            off_task = _clamp01(float(icap_pct.get("Off-task", 0.0)) / 100.0)
            icap_sum = interactive + constructive + active + passive + off_task
            if icap_sum > 0:
                interactive, constructive, active, passive, off_task = (
                    interactive / icap_sum,
                    constructive / icap_sum,
                    active / icap_sum,
                    passive / icap_sum,
                    off_task / icap_sum,
                )
        else:
            interactive = _clamp01(b_active * 0.45 + e_positive * 0.20)
            constructive = _clamp01(b_active * 0.25 + e_neutral * 0.15)
            active = _clamp01(b_active * 0.20 + b_passive * 0.25)
            off_task = _clamp01(b_disruptive * 0.70 + e_frustrated * 0.20)
            passive = _clamp01(1.0 - interactive - constructive - active - off_task)

        behavioral_active.append(round(b_active, 3))
        behavioral_passive.append(round(b_passive, 3))
        behavioral_disruptive.append(round(b_disruptive, 3))
        emotional_positive.append(round(e_positive, 3))
        emotional_neutral.append(round(e_neutral, 3))
        emotional_frustrated.append(round(e_frustrated, 3))
        icap_interactive.append(round(interactive, 3))
        icap_constructive.append(round(constructive, 3))
        icap_active.append(round(active, 3))
        icap_passive.append(round(passive, 3))
        icap_off_task.append(round(off_task, 3))

    points = len(labels)
    behavioral_series = [
        round(behavioral_active[idx] * 0.7 + behavioral_passive[idx] * 0.2 + behavioral_disruptive[idx] * 0.1, 3)
        for idx in range(points)
    ]
    emotional_series = [
        round(emotional_positive[idx] * 0.7 + emotional_neutral[idx] * 0.2 + emotional_frustrated[idx] * 0.1, 3)
        for idx in range(points)
    ]
    cognitive_series = [
        round(
            icap_interactive[idx] * 0.35
            + icap_constructive[idx] * 0.30
            + icap_active[idx] * 0.20
            + icap_passive[idx] * 0.10
            + icap_off_task[idx] * 0.05,
            3,
        )
        for idx in range(points)
    ]

    return {
        "source": "video-analysis",
        "upload_id": upload_id,
        "time_range": time_range,
        "labels": labels,
        "dimension_series": {
            "behavioral": behavioral_series,
            "emotional": emotional_series,
            "cognitive": cognitive_series,
        },
        "behavioral": {
            "Active": behavioral_active,
            "Passive": behavioral_passive,
            "Disruptive": behavioral_disruptive,
        },
        "emotional": {
            "Positive": emotional_positive,
            "Neutral": emotional_neutral,
            "Frustrated": emotional_frustrated,
        },
        "icap": {
            "Interactive": icap_interactive,
            "Constructive": icap_constructive,
            "Active": icap_active,
            "Passive": icap_passive,
            "Off-task": icap_off_task,
        },
    }


def _build_mock_timeline_payload(time_range: str) -> dict[str, object]:
    try:
        start_text, end_text = [part.strip() for part in time_range.split("-")]
        start_sec = _to_seconds(start_text)
        end_sec = _to_seconds(end_text)
    except Exception:
        start_sec = 0
        end_sec = 300

    if end_sec <= start_sec:
        end_sec = start_sec + 300

    step = 30
    point_seconds = _aligned_30s_points(start_sec, end_sec, step=step)
    if not point_seconds:
        first_point = ((start_sec + step - 1) // step) * step
        if first_point <= start_sec:
            first_point += step
        point_seconds = [first_point]
    points = len(point_seconds)
    labels: list[str] = [_to_mmss(val) for val in point_seconds]

    behavioral_active = [max(0.35, 0.62 - idx * 0.015) for idx in range(points)]
    behavioral_passive = [min(0.50, 0.26 + idx * 0.01) for idx in range(points)]
    behavioral_disruptive = [round(1.0 - behavioral_active[idx] - behavioral_passive[idx], 3) for idx in range(points)]

    emotional_positive = [max(0.30, 0.56 - idx * 0.018) for idx in range(points)]
    emotional_neutral = [min(0.55, 0.30 + idx * 0.013) for idx in range(points)]
    emotional_frustrated = [round(1.0 - emotional_positive[idx] - emotional_neutral[idx], 3) for idx in range(points)]

    icap_interactive = [max(0.08, 0.20 - idx * 0.008) for idx in range(points)]
    icap_constructive = [max(0.15, 0.30 - idx * 0.006) for idx in range(points)]
    icap_active = [0.24 for _ in range(points)]
    icap_passive = [min(0.30, 0.16 + idx * 0.01) for idx in range(points)]
    icap_off_task = [
        round(1.0 - icap_interactive[idx] - icap_constructive[idx] - icap_active[idx] - icap_passive[idx], 3)
        for idx in range(points)
    ]

    behavioral_series = [
        round(behavioral_active[idx] * 0.7 + behavioral_passive[idx] * 0.2 + behavioral_disruptive[idx] * 0.1, 3)
        for idx in range(points)
    ]
    emotional_series = [
        round(emotional_positive[idx] * 0.7 + emotional_neutral[idx] * 0.2 + emotional_frustrated[idx] * 0.1, 3)
        for idx in range(points)
    ]
    cognitive_series = [
        round(
            icap_interactive[idx] * 0.35
            + icap_constructive[idx] * 0.30
            + icap_active[idx] * 0.20
            + icap_passive[idx] * 0.10
            + icap_off_task[idx] * 0.05,
            3,
        )
        for idx in range(points)
    ]

    return {
        "time_range": time_range,
        "labels": labels,
        "dimension_series": {
            "behavioral": behavioral_series,
            "emotional": emotional_series,
            "cognitive": cognitive_series,
        },
        "behavioral": {
            "Active": behavioral_active,
            "Passive": behavioral_passive,
            "Disruptive": behavioral_disruptive,
        },
        "emotional": {
            "Positive": emotional_positive,
            "Neutral": emotional_neutral,
            "Frustrated": emotional_frustrated,
        },
        "icap": {
            "Interactive": icap_interactive,
            "Constructive": icap_constructive,
            "Active": icap_active,
            "Passive": icap_passive,
            "Off-task": icap_off_task,
        },
    }


@router.get("/api/metrics")
async def get_metrics() -> dict[str, dict[str, float]]:
    return {
        "behavioral": {
            "Active": 0.52,
            "Passive": 0.36,
            "Disruptive": 0.12,
        },
        "emotional": {
            "Positive": 0.46,
            "Neutral": 0.41,
            "Frustrated": 0.13,
        },
        "icap": {
            "Interactive": 0.18,
            "Constructive": 0.29,
            "Active": 0.23,
            "Passive": 0.20,
            "Off-task": 0.10,
        },
    }


@router.get("/api/metrics_timeline")
async def get_metrics_timeline(
    time_range: str = Query(..., description="时间区间，格式如 05:00-15:00")
) -> dict[str, object]:
    return _build_mock_timeline_payload(time_range)


@router.post("/api/metrics_timeline/video")
async def get_metrics_timeline_from_video(payload: VideoMetricsRequest) -> dict[str, object]:
    response = _build_timeline_from_vision(payload.upload_id, payload.time_range)
    response["subject_name"] = payload.subject_name
    return response
