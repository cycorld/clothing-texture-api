from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from pathlib import Path
from engine.cv import process_image
import uuid

app = FastAPI(
    title="의류 이미지 처리 API",
    description="의류 텍스처를 제품 이미지에 적용하는 API",
    version="1.0.0"
)

# 프로젝트 루트 디렉토리 기준으로 임시 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "temp_uploads"

# 임시 디렉토리가 없으면 생성하고 권한 설정
def setup_upload_dir():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # 디렉토리 권한 설정 (755)
    UPLOAD_DIR.chmod(0o755)

@app.on_event("startup")
async def startup_event():
    setup_upload_dir()

@app.post("/process-image/")
async def process_clothing_image(
    clothing_image: UploadFile = File(...),
    product_name: str = Form(...),
):
    """
    의류 텍스처 이미지와 제품 이미지를 받아서 처리된 결과를 반환합니다.
    """
    temp_files = []
    try:
        # 임시 파일 경로 생성
        temp_id = str(uuid.uuid4())
        clothing_path = UPLOAD_DIR / f"clothing_{temp_id}.jpg"
        product_path = BASE_DIR / f"{product_name}.png"
        output_path = UPLOAD_DIR / f"result_{temp_id}.png"
        
        # temp_files = [clothing_path, product_path, output_path]

        # 업로드된 파일 안전하게 저장
        for file, path in [(clothing_image, clothing_path)]:
            with path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            if not path.exists():
                raise HTTPException(status_code=500, detail=f"파일 저장 실패: {path}")

        # 이미지 처리
        process_image(
            str(clothing_path),
            str(product_path),
            str(output_path),
            debug=True  # 디버깅을 위해 임시로 True로 설정
        )

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="결과 이미지 생성 실패")

        return FileResponse(
            output_path,
            media_type="image/png",
            filename="processed_image.png"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 임시 파일 정리
        pass
        # for path in temp_files:
        #     try:
        #         if path.exists():
        #             path.unlink()
        #     except Exception as e:
        #         print(f"임시 파일 삭제 실패: {path}, 오류: {str(e)}")

@app.get("/")
async def root():
    """API 상태 확인"""
    return {"status": "running", "message": "의류 이미지 처리 API가 정상적으로 실행 중입니다."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 