from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from generator import RussianBabelGenerator
from perplexity import RussianPerplexity
from uuid import uuid4
from fastapi.responses import FileResponse
import json
import os
from typing import List
from typing import Optional, Literal
from cpp_generator import cpp_generate_page  # новый импорт
from semantic_engine import run_semantic_exploration
from fastapi import Query
import numpy as np




temp_storage = {}


app = FastAPI()
generator = RussianBabelGenerator()
pfilter = RussianPerplexity()

class AnalyzeRequest(BaseModel):
    text: str
    window_size: int = 1024
    stride: int = 512
    threshold: float = 50.0

class AnalyzePageRequest(BaseModel):
    hex_id: str
    page_num: int
    window_size: int = 1024
    stride: int = 512
    threshold: float = 50.0

class BatchGenerateRequest(BaseModel):
    hex_id: str
    page_nums: List[int]
    engine: Optional[Literal["cpp", "python"]] = "cpp"  # ← движок, по умолчанию "cpp"

class BatchAnalyzeEntry(BaseModel):
    hex_id: str
    page_num: int

class BatchAnalyzeRequest(BaseModel):
    entries: List[BatchAnalyzeEntry]
    window_size: int = 1024
    stride: int = 512
    threshold: float = 50.0

class BatchAnalyzeBatchRequest(BaseModel):
    hex_id: str
    page_nums: List[int]
    threshold: float = 50.0
    engine: Optional[Literal["cpp", "python"]] = "cpp"



@app.get("/generate")
def generate_page(
    hex_id: str = Query(...),
    page_num: int = Query(...),
    engine: Optional[Literal["cpp", "python"]] = "cpp"  # ← добавляем параметр engine с default="cpp"
):
    if engine == "cpp":
        text = cpp_generate_page(hex_id, page_num)
    else:
        text = generator.generate_page(hex_id, page_num)

    return JSONResponse(content={
        "hex_id": hex_id,
        "page_num": page_num,
        "engine": engine,
        "text": text
    })


@app.post("/generate_batch")
def generate_batch(request: BatchGenerateRequest):
    results = []

    for page_num in request.page_nums:
        if request.engine == "cpp":
            text = cpp_generate_page(request.hex_id, page_num)
        else:
            text = generator.generate_page(request.hex_id, page_num)

        results.append({
            "page_num": page_num,
            "text": text
        })

    return {
        "hex_id": request.hex_id,
        "engine": request.engine,
        "pages": results
    }


@app.post("/analyze")
def analyze_text(request: AnalyzeRequest):
    ppl = pfilter.calculate_perplexity(request.text)
    return JSONResponse(content={
        "perplexity": ppl,
        "text_length": len(request.text)
    })

@app.post("/analyze_window")
def analyze_with_windows(request: AnalyzeRequest):
    results = pfilter.sliding_window_analysis(
        text=request.text,
        window_size=request.window_size,
        stride=request.stride,
        threshold=request.threshold
    )
    return {"windows": results}

@app.post("/analyze_page_windowed")
def analyze_page_windowed(request: AnalyzePageRequest):
    text = generator.generate_page(request.hex_id, request.page_num)

    results = pfilter.sliding_window_analysis(
        text=text,
        window_size=request.window_size,
        stride=request.stride,
        threshold=request.threshold
    )

    # Проверим: есть ли окна с perplexity выше порога?
    exceeded = any(fragment["perplexity"] > request.threshold for fragment in results)

    response = {
        "hex_id": request.hex_id,
        "page_num": request.page_num,
        "fragments": results
    }

    if exceeded:
        # Генерируем уникальный ID
        analysis_id = str(uuid4())
        temp_storage[analysis_id] = response

        response["download_url"] = f"/save_page_analysis?analysis_id={analysis_id}"

    return response

@app.post("/batch_analyze_batch")
def batch_analyze_batch(request: BatchAnalyzeBatchRequest):
    results = []

    for page_num in request.page_nums:
        if request.engine == "cpp":
            text = cpp_generate_page(request.hex_id, page_num)
        else:
            text = generator.generate_page(request.hex_id, page_num)

        perplexity = pfilter.calculate_perplexity(text)

        if perplexity <= request.threshold:
            results.append({
                "page_num": page_num, 
                "perplexity": perplexity,
                "text": text
            })

    return {
        "hex_id": request.hex_id,
        "engine": request.engine,
        "threshold": request.threshold,
        "found": results
    }


@app.post("/batch_analyze_windowed")
def batch_analyze_windowed(request: BatchAnalyzeRequest):
    results = []

    for entry in request.entries:
        text = generator.generate_page(entry.hex_id, entry.page_num)
        fragments = pfilter.sliding_window_analysis(
            text=text,
            window_size=request.window_size,
            stride=request.stride,
            threshold=request.threshold
        )

        analysis = {
            "hex_id": entry.hex_id,
            "page_num": entry.page_num,
            "fragments": fragments
        }

        # Проверка на превышение порога
        exceeded = any(f["perplexity"] > request.threshold for f in fragments)
        if exceeded:
            analysis_id = str(uuid4())
            temp_storage[analysis_id] = analysis
            analysis["download_url"] = f"/save_page_analysis?analysis_id={analysis_id}"

        results.append(analysis)

    return {"results": results}


@app.get("/save_page_analysis")
def save_page_analysis(analysis_id: str):
    if analysis_id not in temp_storage:
        return JSONResponse(status_code=404, content={"error": "Analysis not found"})

    data = temp_storage[analysis_id]
    filename = f"analysis_{data['hex_id']}_{data['page_num']}.json"
    filepath = os.path.join("results", filename)

    os.makedirs("results", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return FileResponse(filepath, media_type="application/json", filename=filename)

@app.get("/semantic_explore")
def semantic_explore(
    num_hex: int = Query(10, description="Количество hex_id для генерации"),
    pages_per_hex: int = Query(5, description="Сколько страниц на один hex_id"),
    threshold: float = Query(0.9, description="Порог косинусного сходства")
):
    matches = run_semantic_exploration(
        num_hex=num_hex,
        pages_per_hex=pages_per_hex,
        similarity_threshold=threshold
    )

    # Конвертер для np.float32 → float
    def convert_np(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    if not matches:
        return {
            "status": "ok",
            "matches_found": 0,
            "file": None,
            "message": "Совпадений не найдено."
        }

    # Сохраняем только если есть совпадения
    os.makedirs("semantic_results", exist_ok=True)
    filename = f"semantic_{str(uuid4())[:8]}.json"
    filepath = os.path.join("semantic_results", filename)

    with open(filepath, "w", encoding="utf-8") as fout:
        for match in matches:
            json.dump(match, fout, ensure_ascii=False, default=convert_np)
            fout.write("\n")

    return {
        "status": "ok",
        "matches_found": len(matches),
        "file": f"/semantic_results/{filename}"
    }

@app.get("/")
def read_root():
    return{"welcome, user"}





