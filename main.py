"""CerviCa clinical report API — stateless 3D scan analysis."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from watch_scans import analyze_stl

ALLOWED_SUFFIXES = {".stl", ".ply", ".obj"}
STL_SUFFIXES = {".stl"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _cors_origins() -> list[str]:
    extra = os.getenv("CERVICA_CORS_ORIGINS", "")
    origins = list(DEFAULT_CORS_ORIGINS)
    for item in extra.split(","):
        origin = item.strip()
        if origin and origin not in origins:
            origins.append(origin)
    return origins


app = FastAPI(
    title="CerviCa Report API",
    description="3D 스캔을 실시간 분석한 뒤 PDF만 반환하고, 서버 디스크의 환자 파일은 즉시 파기합니다.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _remove_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.is_file():
            os.remove(path)
    except OSError:
        pass


def _purge_work_dir(work_dir: Path | None) -> None:
    if work_dir is None:
        return
    shutil.rmtree(work_dir, ignore_errors=True)


def _report_to_pdf_bytes(png_path: Path) -> bytes:
    buffer = BytesIO()
    with Image.open(png_path) as image:
        image.convert("RGB").save(buffer, "PDF", resolution=160.0)
    return buffer.getvalue()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate-report")
async def generate_report(file: UploadFile = File(...)) -> StreamingResponse:
    original_name = Path(file.filename or "scan.stl").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 파일 형식입니다. .stl, .ply, .obj 만 업로드할 수 있습니다.",
        )
    if suffix not in STL_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="현재 분석기는 binary STL(.stl)만 지원합니다.",
        )

    work_dir: Path | None = None
    scan_path: Path | None = None
    report_png: Path | None = None
    pdf_bytes = b""

    try:
        work_dir = Path(tempfile.mkdtemp(prefix=f"cervica_{uuid.uuid4().hex}_"))
        scan_path = work_dir / f"scan{suffix}"

        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="빈 파일은 분석할 수 없습니다.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="파일 용량이 250MB를 초과합니다.")
        scan_path.write_bytes(payload)
        del payload

        png_result = analyze_stl(str(scan_path))
        _remove_file(scan_path)
        scan_path = None

        if png_result is None or not Path(png_result).exists():
            raise HTTPException(
                status_code=422,
                detail="3D 스캔에서 얼굴을 분석하지 못했습니다. 정면 두부 스캔인지 확인해 주세요.",
            )

        report_png = Path(png_result)
        pdf_bytes = _report_to_pdf_bytes(report_png)
        _remove_file(report_png)
        report_png = None

        if not pdf_bytes:
            raise HTTPException(status_code=500, detail="PDF 리포트 생성에 실패했습니다.")

        filename = f"{Path(original_name).stem}_clinical_report.pdf"
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {exc}") from exc
    finally:
        _remove_file(scan_path)
        _remove_file(report_png)
        _purge_work_dir(work_dir)
        await file.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
