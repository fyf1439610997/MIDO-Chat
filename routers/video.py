import json
import math
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from opencc import OpenCC
from pydantic import BaseModel, Field

from llm_config import build_asr_client, build_llm_client, build_speech_client, get_asr_model
from prompt_templates import render_prompt

router = APIRouter(prefix="/api/video", tags=["video"])

MAX_VIDEO_SIZE_BYTES = 10 * 1024 * 1024 * 1024
ASR_CHUNK_SECONDS = 10
WORK_DIR = Path("generated/video")
UPLOAD_DIR = WORK_DIR / "uploads"
VIDEO_DIR = WORK_DIR / "videos"
AUDIO_DIR = WORK_DIR / "audio"
TRANSCRIPT_DIR = WORK_DIR / "transcripts"
CHUNK_DIR = WORK_DIR / "chunks"
VISION_METRICS_DIR = WORK_DIR / "vision_metrics"
VISION_PROJECT_DIR = Path("RT-DETR-DHSA")
VISION_SCRIPT_PATH = VISION_PROJECT_DIR / "classroom_dual_model_sampler.py"
ICAP_WAIT_TIMEOUT_SECONDS = 2 * 60 * 60

for path in (UPLOAD_DIR, VIDEO_DIR, AUDIO_DIR, TRANSCRIPT_DIR, CHUNK_DIR, VISION_METRICS_DIR):
    path.mkdir(parents=True, exist_ok=True)


class UploadInitPayload(BaseModel):
    filename: str = Field(..., min_length=1)
    file_size: int = Field(..., gt=0)
    total_chunks: int = Field(..., gt=0)
    chunk_size: int = Field(..., gt=0)


class UploadCompletePayload(BaseModel):
    upload_id: str = Field(..., min_length=1)


class ExtractAudioPayload(BaseModel):
    upload_id: str = Field(..., min_length=1)
    output_format: str = Field(default="mp3")


class TranscribeAudioPayload(BaseModel):
    upload_id: str = Field(..., min_length=1)


def _sanitize_folder_name(filename: str) -> str:
    stem = Path(filename).stem.strip()
    if not stem:
        stem = "upload"
    kept: list[str] = []
    for ch in stem:
        code = ord(ch)
        is_cjk = 0x4E00 <= code <= 0x9FFF
        if ch.isalnum() or ch in {"-", "_"} or is_cjk:
            kept.append(ch)
        else:
            kept.append("-")
    safe = "".join(kept).strip("-_.")
    return safe or "upload"


def _find_upload_folder(upload_id: str) -> Path | None:
    # Backward compatible: older runs used upload_id as folder name.
    legacy = UPLOAD_DIR / upload_id
    if legacy.exists():
        return legacy
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
            return folder
    return None


def _meta_file(upload_id: str, meta: dict | None = None) -> Path:
    folder = _find_upload_folder(upload_id)
    if folder is None and meta is not None:
        folder_name = str(meta.get("upload_folder", upload_id))
        folder = UPLOAD_DIR / folder_name
    if folder is None:
        folder = UPLOAD_DIR / upload_id
    return folder / "meta.json"


def _read_meta(upload_id: str) -> dict:
    meta_path = _meta_file(upload_id)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="上传任务不存在，请重新上传视频。")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_meta(upload_id: str, meta: dict) -> None:
    upload_path = _meta_file(upload_id, meta).parent
    upload_path.mkdir(parents=True, exist_ok=True)
    with _meta_file(upload_id, meta).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _vision_download_url(upload_id: str, file_type: str) -> str:
    return f"/api/video/vision_metrics/download/{upload_id}/{file_type}"


def _to_transcription_dict(result: object) -> dict:
    if hasattr(result, "model_dump"):
        raw_data = result.model_dump()
    elif isinstance(result, dict):
        raw_data = result
    else:
        raw_data = {"text": str(getattr(result, "text", "")), "segments": []}
    return raw_data if isinstance(raw_data, dict) else {"text": "", "segments": []}


def _transcribe_audio_file(client: object, asr_model: str, audio_path: Path) -> dict:
    with audio_path.open("rb") as audio_file:
        try:
            result = client.audio.transcriptions.create(
                model=asr_model,
                file=audio_file,
                response_format="json",
                language="zh",
                prompt="请使用简体中文逐字转写课堂语音。",
            )
        except Exception:
            audio_file.seek(0)
            result = client.audio.transcriptions.create(
                model=asr_model,
                file=audio_file,
                response_format="json",
            )
    return _to_transcription_dict(result)


def _transcribe_with_speech_settings() -> tuple[dict | None, str]:
    speech_client, speech_settings = build_speech_client()
    speech_model = speech_settings.get("model", "").strip()
    if not speech_client or not speech_model:
        return None, ""
    return {"client": speech_client, "model": speech_model, "settings": speech_settings}, speech_model


def _split_audio_into_chunks(upload_id: str, audio_path: Path) -> list[Path]:
    meta = _read_meta(upload_id)
    folder_name = str(meta.get("upload_folder", upload_id))
    chunk_folder = CHUNK_DIR / folder_name
    if chunk_folder.exists():
        shutil.rmtree(chunk_folder, ignore_errors=True)
    chunk_folder.mkdir(parents=True, exist_ok=True)

    chunk_pattern = str(chunk_folder / "chunk-%04d.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(ASR_CHUNK_SECONDS),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        chunk_pattern,
    ]
    subprocess.run(command, check=True, capture_output=True)
    return sorted(chunk_folder.glob("chunk-*.wav"))


def _get_audio_duration_seconds(audio_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        return float(result.stdout.strip() or "0")
    except ValueError:
        return 0.0


def _snap_to_10s_floor(value: float) -> int:
    return int((value // ASR_CHUNK_SECONDS) * ASR_CHUNK_SECONDS)


def _snap_to_10s_ceil(value: float) -> int:
    if value <= 0:
        return ASR_CHUNK_SECONDS
    return int(((value + ASR_CHUNK_SECONDS - 1e-9) // ASR_CHUNK_SECONDS) * ASR_CHUNK_SECONDS)


_opencc_t2s = OpenCC("t2s")


def _to_simplified(text: str) -> str:
    if not text:
        return ""
    return _opencc_t2s.convert(text)


def _normalize_icap_percentages(raw: dict) -> dict[str, float]:
    keys = ["Interactive", "Constructive", "Active", "Passive", "Off-task"]
    values = [max(0.0, float(raw.get(k, 0.0) or 0.0)) for k in keys]
    total = sum(values)
    if total <= 0:
        values = [10.0, 15.0, 25.0, 40.0, 10.0]
        total = 100.0
    scaled = [round(v * 100.0 / total, 2) for v in values]
    diff = round(100.0 - sum(scaled), 2)
    if scaled:
        max_idx = max(range(len(scaled)), key=lambda i: scaled[i])
        scaled[max_idx] = round(scaled[max_idx] + diff, 2)
    return {keys[i]: scaled[i] for i in range(len(keys))}


def _icap_fallback_for_text(text: str) -> dict[str, float]:
    t = (text or "").strip()
    if not t:
        return {"Interactive": 5.0, "Constructive": 10.0, "Active": 20.0, "Passive": 55.0, "Off-task": 10.0}
    length = len(t)
    q_count = t.count("？") + t.count("?")
    interactive = min(35.0, 8.0 + q_count * 2.0)
    constructive = min(35.0, 12.0 + (length / 100.0) * 4.0)
    active = 22.0
    off_task = 8.0
    passive = max(0.0, 100.0 - interactive - constructive - active - off_task)
    return _normalize_icap_percentages(
        {
            "Interactive": interactive,
            "Constructive": constructive,
            "Active": active,
            "Passive": passive,
            "Off-task": off_task,
        }
    )


def _extract_json_candidate(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def _infer_icap_with_llm(window_payloads: list[dict]) -> dict[int, dict[str, float]]:
    client, model, _provider = build_llm_client()
    if not client:
        return {
            int(item["window_index"]): _icap_fallback_for_text(item.get("transcript_text", ""))
            for item in window_payloads
        }

    system_prompt = render_prompt("icap_window_system")
    user_prompt = render_prompt(
        "icap_window_user",
        window_payloads_json=json.dumps({"windows": window_payloads}, ensure_ascii=False, indent=2),
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content if resp.choices else ""
        obj = json.loads(_extract_json_candidate(content))
        rows = obj.get("windows", []) if isinstance(obj, dict) else []
    except Exception:
        rows = []

    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("window_index"))
        except Exception:
            continue
        result[idx] = _normalize_icap_percentages(row)

    for item in window_payloads:
        idx = int(item["window_index"])
        if idx not in result:
            result[idx] = _icap_fallback_for_text(item.get("transcript_text", ""))
    return result


def _merge_icap_into_metrics(upload_id: str, icap_by_window: dict[int, dict[str, float]]) -> None:
    meta = _read_meta(upload_id)
    json_path = Path(str(meta.get("vision_json_path", "")).strip())
    csv_path = Path(str(meta.get("vision_csv_path", "")).strip())
    if not json_path.exists():
        raise RuntimeError("视觉统计JSON不存在，无法写入ICAP。")

    with json_path.open("r", encoding="utf-8") as f:
        windows = json.load(f)
    if not isinstance(windows, list):
        raise RuntimeError("视觉统计JSON格式无效。")

    for item in windows:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("window_index", -1) or -1)
        icap = icap_by_window.get(idx, _icap_fallback_for_text(""))
        item["icap_percentages"] = icap

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(windows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "window_index",
        "start_second",
        "end_second",
        "sampled_frames",
        "detected_person_boxes",
        "behavior_percentages_json",
        "expression_percentages_json",
        "icap_percentages_json",
    ]
    rows = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "window_index": item.get("window_index", 0),
                "start_second": item.get("start_second", 0),
                "end_second": item.get("end_second", 0),
                "sampled_frames": item.get("sampled_frames", 0),
                "detected_person_boxes": item.get("detected_person_boxes", 0),
                "behavior_percentages_json": json.dumps(item.get("behavior_percentages", {}), ensure_ascii=False),
                "expression_percentages_json": json.dumps(item.get("expression_percentages", {}), ensure_ascii=False),
                "icap_percentages_json": json.dumps(item.get("icap_percentages", {}), ensure_ascii=False),
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_icap_analysis(upload_id: str) -> None:
    try:
        meta = _read_meta(upload_id)
    except HTTPException:
        return

    start_wait = time.time()
    while True:
        meta = _read_meta(upload_id)
        status = str(meta.get("vision_status", "idle"))
        if status == "completed":
            break
        if status == "failed":
            meta["icap_status"] = "failed"
            meta["icap_error"] = "ICAP分析取消：视觉分析失败。"
            _save_meta(upload_id, meta)
            return
        if time.time() - start_wait > ICAP_WAIT_TIMEOUT_SECONDS:
            meta["icap_status"] = "failed"
            meta["icap_error"] = "ICAP分析超时：等待视觉分析完成超时。"
            _save_meta(upload_id, meta)
            return
        time.sleep(3)

    transcript_path_raw = str(meta.get("transcript_path", "")).strip()
    if not transcript_path_raw:
        meta["icap_status"] = "failed"
        meta["icap_error"] = "ICAP分析失败：未找到转写文件。"
        _save_meta(upload_id, meta)
        return
    transcript_path = Path(transcript_path_raw)
    if not transcript_path.exists():
        meta["icap_status"] = "failed"
        meta["icap_error"] = "ICAP分析失败：转写文件不存在。"
        _save_meta(upload_id, meta)
        return

    meta["icap_status"] = "processing"
    meta["icap_error"] = ""
    _save_meta(upload_id, meta)

    try:
        with transcript_path.open("r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        segments = transcript_data.get("segments", []) if isinstance(transcript_data, dict) else []

        vision_json_path = Path(str(meta.get("vision_json_path", "")).strip())
        with vision_json_path.open("r", encoding="utf-8") as f:
            windows = json.load(f)
        if not isinstance(windows, list):
            raise RuntimeError("视觉统计窗口数据格式无效。")

        payloads: list[dict] = []
        for item in windows:
            if not isinstance(item, dict):
                continue
            ws = float(item.get("start_second", 0.0) or 0.0)
            we = float(item.get("end_second", 0.0) or 0.0)
            texts: list[str] = []
            for seg in segments if isinstance(segments, list) else []:
                if not isinstance(seg, dict):
                    continue
                ss = float(seg.get("start", 0.0) or 0.0)
                se = float(seg.get("end", 0.0) or 0.0)
                if se > ws and ss < we:
                    txt = str(seg.get("text", "")).strip()
                    if txt:
                        texts.append(txt)
            payloads.append(
                {
                    "window_index": int(item.get("window_index", 0) or 0),
                    "start_second": ws,
                    "end_second": we,
                    "transcript_text": "\n".join(texts),
                }
            )

        icap_by_window = _infer_icap_with_llm(payloads)
        _merge_icap_into_metrics(upload_id, icap_by_window)
        meta = _read_meta(upload_id)
        meta["icap_status"] = "completed"
        meta["icap_error"] = ""
        _save_meta(upload_id, meta)
    except Exception as exc:
        meta = _read_meta(upload_id)
        meta["icap_status"] = "failed"
        meta["icap_error"] = f"ICAP分析失败：{exc}"
        _save_meta(upload_id, meta)


def _run_vision_analysis(upload_id: str, video_path_raw: str) -> None:
    try:
        meta = _read_meta(upload_id)
    except HTTPException:
        return

    meta["vision_status"] = "processing"
    meta["vision_error"] = ""
    meta["vision_json_path"] = ""
    meta["vision_csv_path"] = ""
    meta["vision_percent"] = 0.0
    meta["vision_current"] = 0
    meta["vision_total"] = 0
    _save_meta(upload_id, meta)

    if not VISION_SCRIPT_PATH.exists():
        meta["vision_status"] = "failed"
        meta["vision_error"] = f"未找到视觉分析脚本：{VISION_SCRIPT_PATH}"
        _save_meta(upload_id, meta)
        return

    video_path = Path(video_path_raw)
    folder_name = str(meta.get("upload_folder", upload_id))
    output_dir = VISION_METRICS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    command = [
        "python",
        str(VISION_SCRIPT_PATH.name),
        "--video",
        str(video_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--device",
        "0",
        "--progress-file",
        str(progress_path.resolve()),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(VISION_PROJECT_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        meta["vision_status"] = "failed"
        meta["vision_error"] = f"视觉分析启动失败：{exc}"
        _save_meta(upload_id, meta)
        return

    json_path = output_dir / "classroom_30s_stats.json"
    csv_path = output_dir / "classroom_30s_stats.csv"
    if result.returncode != 0 or not json_path.exists():
        stderr_tail = (result.stderr or "").strip()[-500:]
        stdout_tail = (result.stdout or "").strip()[-500:]
        detail = stderr_tail or stdout_tail or f"退出码 {result.returncode}"
        meta["vision_status"] = "failed"
        meta["vision_error"] = f"视觉分析失败：{detail}"
        meta["vision_progress_path"] = str(progress_path)
        _save_meta(upload_id, meta)
        return

    meta["vision_status"] = "completed"
    meta["vision_error"] = ""
    meta["vision_json_path"] = str(json_path)
    meta["vision_csv_path"] = str(csv_path)
    meta["vision_progress_path"] = str(progress_path)
    meta["vision_percent"] = 100.0
    _save_meta(upload_id, meta)


@router.get("/limits")
async def get_video_limits() -> dict[str, int]:
    return {"max_video_size_bytes": MAX_VIDEO_SIZE_BYTES}


@router.post("/upload/init")
async def init_upload(payload: UploadInitPayload) -> dict[str, str | int]:
    if payload.file_size > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"视频过大：当前上限为 {MAX_VIDEO_SIZE_BYTES // (1024**3)}GB，请压缩后重试。",
        )

    upload_id = uuid.uuid4().hex
    safe_filename = Path(payload.filename).name
    base_folder = _sanitize_folder_name(safe_filename)
    upload_folder = UPLOAD_DIR / base_folder
    suffix = 2
    while upload_folder.exists():
        upload_folder = UPLOAD_DIR / f"{base_folder}-{suffix}"
        suffix += 1
    meta = {
        "upload_id": upload_id,
        "upload_folder": upload_folder.name,
        "filename": safe_filename,
        "file_size": payload.file_size,
        "total_chunks": payload.total_chunks,
        "chunk_size": payload.chunk_size,
        "video_path": "",
        "audio_path": "",
        "transcript_path": "",
    }
    _save_meta(upload_id, meta)
    return {
        "upload_id": upload_id,
        "filename": safe_filename,
        "max_video_size_bytes": MAX_VIDEO_SIZE_BYTES,
    }


@router.post("/upload/chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
) -> dict[str, str | int]:
    meta = _read_meta(upload_id)
    total_chunks = int(meta["total_chunks"])
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="分片索引非法。")

    upload_folder = _find_upload_folder(upload_id)
    if upload_folder is None:
        raise HTTPException(status_code=404, detail="上传任务不存在，请重新上传视频。")
    chunk_path = upload_folder / f"{chunk_index:08d}.part"
    with chunk_path.open("wb") as out:
        shutil.copyfileobj(chunk.file, out)
    return {"upload_id": upload_id, "chunk_index": chunk_index, "status": "ok"}


@router.post("/upload/complete")
async def complete_upload(payload: UploadCompletePayload, background_tasks: BackgroundTasks) -> dict[str, str | int]:
    meta = _read_meta(payload.upload_id)
    total_chunks = int(meta["total_chunks"])
    upload_path = _find_upload_folder(payload.upload_id)
    if upload_path is None:
        raise HTTPException(status_code=404, detail="上传任务不存在，请重新上传视频。")

    missing = []
    for idx in range(total_chunks):
        if not (upload_path / f"{idx:08d}.part").exists():
            missing.append(idx)
    if missing:
        raise HTTPException(status_code=400, detail=f"分片缺失：{len(missing)} 个，请重新上传。")

    merged_name = f"{upload_path.name}-{meta['filename']}"
    merged_path = VIDEO_DIR / merged_name
    with merged_path.open("wb") as merged:
        for idx in range(total_chunks):
            part = upload_path / f"{idx:08d}.part"
            with part.open("rb") as src:
                shutil.copyfileobj(src, merged)

    meta["video_path"] = str(merged_path)
    meta["vision_status"] = "queued"
    meta["vision_error"] = ""
    meta["vision_json_path"] = ""
    meta["vision_csv_path"] = ""
    meta["vision_progress_path"] = ""
    meta["vision_percent"] = 0.0
    meta["vision_current"] = 0
    meta["vision_total"] = 0
    meta["icap_status"] = "idle"
    meta["icap_error"] = ""
    _save_meta(payload.upload_id, meta)
    background_tasks.add_task(_run_vision_analysis, payload.upload_id, str(merged_path))

    return {
        "upload_id": payload.upload_id,
        "video_filename": merged_name,
        "video_size_bytes": merged_path.stat().st_size,
        "vision_status": "queued",
    }


@router.post("/extract_audio")
async def extract_audio(payload: ExtractAudioPayload) -> dict[str, object]:
    meta = _read_meta(payload.upload_id)
    video_path_raw = meta.get("video_path", "")
    if not video_path_raw:
        raise HTTPException(status_code=400, detail="请先完成视频上传。")
    video_path = Path(video_path_raw)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在，请重新上传。")

    output_format = payload.output_format.lower()
    if output_format not in {"mp3", "wav"}:
        raise HTTPException(status_code=400, detail="仅支持 mp3 或 wav。")

    upload_folder = _find_upload_folder(payload.upload_id)
    readable_name = upload_folder.name if upload_folder else str(meta.get("upload_folder", payload.upload_id))
    audio_name = f"{readable_name}.{output_format}"
    audio_path = AUDIO_DIR / audio_name

    if output_format == "mp3":
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(audio_path),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]

    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="服务器未安装 ffmpeg，无法分离音频。") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")[-300:]
        raise HTTPException(status_code=500, detail=f"音频分离失败：{stderr or 'ffmpeg 执行失败'}") from exc

    meta["audio_path"] = str(audio_path)
    meta["transcribe_status"] = "idle"
    meta["transcribe_current"] = 0
    meta["transcribe_total"] = 0
    _save_meta(payload.upload_id, meta)
    return {
        "upload_id": payload.upload_id,
        "audio_filename": audio_name,
        "download_url": f"/api/video/audio/download/{audio_name}",
        "progress": {"current": 1, "total": 1},
    }


@router.post("/transcribe_audio")
async def transcribe_audio(payload: TranscribeAudioPayload, background_tasks: BackgroundTasks) -> dict:
    meta = _read_meta(payload.upload_id)
    audio_path_raw = meta.get("audio_path", "")
    if not audio_path_raw:
        raise HTTPException(status_code=400, detail="请先完成音频分离。")
    audio_path = Path(audio_path_raw)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在，请重新分离。")

    # 优先使用 speech_* 配置（可用于阿里语音等专用服务），未配置则回退到当前 asr 配置
    speech_bundle, speech_model = _transcribe_with_speech_settings()
    if speech_bundle:
        client = speech_bundle["client"]
        asr_model = speech_model
    else:
        client, _provider = build_asr_client()
        if not client:
            raise HTTPException(status_code=400, detail="未配置可用模型密钥，无法执行音频转写。")
        asr_model = get_asr_model()

    upload_folder = _find_upload_folder(payload.upload_id)
    folder_name = upload_folder.name if upload_folder else payload.upload_id
    chunk_folder = CHUNK_DIR / folder_name
    transcribe_total = 1
    transcribe_current = 0
    try:
        duration_sec = _get_audio_duration_seconds(audio_path)
        estimated_total = max(1, math.ceil(duration_sec / ASR_CHUNK_SECONDS))
        meta["transcribe_status"] = "processing"
        meta["transcribe_current"] = 0
        meta["transcribe_total"] = estimated_total
        _save_meta(payload.upload_id, meta)

        chunk_files = _split_audio_into_chunks(payload.upload_id, audio_path)
        if not chunk_files:
            raise ValueError("分块结果为空。")
        transcribe_total = len(chunk_files)
        meta["transcribe_total"] = transcribe_total
        _save_meta(payload.upload_id, meta)

        merged_text_parts: list[str] = []
        merged_segments: list[dict] = []
        for index, chunk_file in enumerate(chunk_files):
            chunk_result = _transcribe_audio_file(client, asr_model, chunk_file)
            text_part = _to_simplified(str(chunk_result.get("text", "")).strip())
            if text_part:
                merged_text_parts.append(text_part)
                base_offset = index * ASR_CHUNK_SECONDS
                merged_segments.append(
                    {
                        "start": base_offset,
                        "end": base_offset + ASR_CHUNK_SECONDS,
                        "text": text_part,
                    }
                )
            transcribe_current = index + 1
            meta["transcribe_current"] = transcribe_current
            _save_meta(payload.upload_id, meta)

        raw_data = {
            "text": "\n".join(part for part in merged_text_parts if part),
            "segments": merged_segments,
        }
    except FileNotFoundError as exc:
        meta["transcribe_status"] = "failed"
        _save_meta(payload.upload_id, meta)
        raise HTTPException(status_code=500, detail="服务器未安装 ffmpeg/ffprobe，无法执行分块转写。") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")[-300:]
        meta["transcribe_status"] = "failed"
        _save_meta(payload.upload_id, meta)
        raise HTTPException(status_code=500, detail=f"音频处理失败：{stderr or 'ffmpeg/ffprobe 执行失败'}") from exc
    except Exception as exc:
        meta["transcribe_status"] = "failed"
        _save_meta(payload.upload_id, meta)
        raise HTTPException(status_code=500, detail=f"音频转写失败：{exc}") from exc
    finally:
        if chunk_folder.exists():
            shutil.rmtree(chunk_folder, ignore_errors=True)

    segments = raw_data.get("segments", [])
    transcript_text = _to_simplified(str(raw_data.get("text", "")).strip())
    if isinstance(segments, list):
        normalized_segments: list[dict] = []
        for idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            raw_start = float(segment.get("start", 0.0) or 0.0)
            raw_end = float(segment.get("end", 0.0) or 0.0)
            snapped_start = _snap_to_10s_floor(raw_start)
            snapped_end = _snap_to_10s_ceil(raw_end)
            if snapped_end <= snapped_start:
                snapped_end = snapped_start + ASR_CHUNK_SECONDS
            normalized_segments.append(
                {
                    "speaker": str(segment.get("speaker", f"SPK{idx + 1}")),
                    "start": snapped_start,
                    "end": snapped_end,
                    "text": _to_simplified(str(segment.get("text", "")).strip()),
                }
            )
        segments = normalized_segments

    transcript_payload = {
        "upload_id": payload.upload_id,
        "model": asr_model,
        "text": transcript_text,
        "segments": segments,
    }
    readable_name = upload_folder.name if upload_folder else str(meta.get("upload_folder", payload.upload_id))
    transcript_name = f"{readable_name}-transcript.json"
    transcript_path = TRANSCRIPT_DIR / transcript_name
    with transcript_path.open("w", encoding="utf-8") as f:
        json.dump(transcript_payload, f, ensure_ascii=False, indent=2)

    meta["transcript_path"] = str(transcript_path)
    meta["transcribe_status"] = "completed"
    meta["transcribe_current"] = transcribe_total
    meta["transcribe_total"] = transcribe_total
    meta["icap_status"] = "queued"
    meta["icap_error"] = ""
    _save_meta(payload.upload_id, meta)
    background_tasks.add_task(_run_icap_analysis, payload.upload_id)
    return {
        "upload_id": payload.upload_id,
        "model": asr_model,
        "text": transcript_text,
        "segments": segments,
        "transcript_filename": transcript_name,
        "download_url": f"/api/video/transcript/download/{transcript_name}",
        "progress": {"current": transcribe_current, "total": transcribe_total},
    }


@router.get("/transcribe_progress/{upload_id}")
async def get_transcribe_progress(upload_id: str) -> dict[str, int | str]:
    meta = _read_meta(upload_id)
    return {
        "upload_id": upload_id,
        "status": str(meta.get("transcribe_status", "idle")),
        "current": int(meta.get("transcribe_current", 0) or 0),
        "total": int(meta.get("transcribe_total", 0) or 0),
    }


@router.get("/vision_progress/{upload_id}")
async def get_vision_progress(upload_id: str) -> dict[str, str | float | int]:
    meta = _read_meta(upload_id)
    progress_path_raw = str(meta.get("vision_progress_path", "")).strip()
    progress_payload = None
    if progress_path_raw:
        progress_path = Path(progress_path_raw)
        if progress_path.exists():
            try:
                with progress_path.open("r", encoding="utf-8") as f:
                    progress_payload = json.load(f)
            except Exception:
                progress_payload = None
    if isinstance(progress_payload, dict):
        meta["vision_current"] = int(progress_payload.get("current_sample_points", 0) or 0)
        meta["vision_total"] = int(progress_payload.get("total_sample_points", 0) or 0)
        meta["vision_percent"] = float(progress_payload.get("percent", 0.0) or 0.0)
        _save_meta(upload_id, meta)
    if str(meta.get("vision_status", "")) == "completed" and float(meta.get("vision_percent", 0.0) or 0.0) < 100.0:
        meta["vision_percent"] = 100.0
        _save_meta(upload_id, meta)
    return {
        "upload_id": upload_id,
        "status": str(meta.get("vision_status", "idle")),
        "error": str(meta.get("vision_error", "")),
        "json_path": str(meta.get("vision_json_path", "")),
        "csv_path": str(meta.get("vision_csv_path", "")),
        "json_download_url": _vision_download_url(upload_id, "json"),
        "csv_download_url": _vision_download_url(upload_id, "csv"),
        "percent": float(meta.get("vision_percent", 0.0) or 0.0),
        "current": int(meta.get("vision_current", 0) or 0),
        "total": int(meta.get("vision_total", 0) or 0),
        "icap_status": str(meta.get("icap_status", "idle")),
        "icap_error": str(meta.get("icap_error", "")),
    }


@router.get("/audio/download/{audio_filename}")
async def download_audio(audio_filename: str) -> FileResponse:
    safe_name = Path(audio_filename).name
    target = AUDIO_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在。")
    media_type = "audio/mpeg" if target.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(path=target, media_type=media_type, filename=safe_name)


@router.get("/transcript/download/{transcript_filename}")
async def download_transcript(transcript_filename: str) -> FileResponse:
    safe_name = Path(transcript_filename).name
    target = TRANSCRIPT_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="转写文件不存在。")
    return FileResponse(path=target, media_type="application/json", filename=safe_name)


@router.get("/vision_metrics/download/{upload_id}/{file_type}")
async def download_vision_metrics(upload_id: str, file_type: str) -> FileResponse:
    meta = _read_meta(upload_id)
    ft = file_type.lower().strip()
    if ft == "json":
        path_raw = str(meta.get("vision_json_path", "")).strip()
        media_type = "application/json"
    elif ft == "csv":
        path_raw = str(meta.get("vision_csv_path", "")).strip()
        media_type = "text/csv"
    else:
        raise HTTPException(status_code=400, detail="file_type 仅支持 json 或 csv。")
    if not path_raw:
        raise HTTPException(status_code=404, detail="该上传任务尚未产出对应三维投入度文件。")
    target = Path(path_raw)
    if not target.exists():
        raise HTTPException(status_code=404, detail="三维投入度文件不存在。")
    return FileResponse(path=target, media_type=media_type, filename=target.name)
