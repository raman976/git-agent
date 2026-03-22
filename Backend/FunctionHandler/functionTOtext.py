import os
import sys
import importlib.util
import threading
from typing import Any, Dict, List, Optional

import numpy as np


CURRENT_DIR = os.path.dirname(__file__)
REPO_MANAGER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "RepoManager"))


def load_function_extractor_module() -> Any:
    extractor_path = os.path.join(REPO_MANAGER_DIR, "functionExtractor.py")
    if REPO_MANAGER_DIR not in sys.path:
        sys.path.insert(0, REPO_MANAGER_DIR)

    spec = importlib.util.spec_from_file_location("functionExtractor", extractor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load functionExtractor from {extractor_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_EMBEDDING_MODELS: Dict[str, Any] = {}
_EMBEDDING_MODELS_LOCK = threading.Lock()


def function_to_text(func: Dict[str, object]) -> str:
    language = func.get("language", "unknown")
    start_line = func.get("start_line", "?")
    end_line = func.get("end_line", "?")

    return (
        f"Function: {func.get('name', '')}\n"
        f"Language: {language}\n"
        f"File: {func.get('file', '')}:{start_line}-{end_line}\n\n"
        f"Code:\n{func.get('code', '')}"
    )


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc

    with _EMBEDDING_MODELS_LOCK:
        cached = _EMBEDDING_MODELS.get(model_name)
        if cached is not None:
            return cached

        model = SentenceTransformer(model_name)
        _EMBEDDING_MODELS[model_name] = model
        return model


def build_function_documents(functions: List[Dict[str, object]]) -> List[str]:
    return [function_to_text(func) for func in functions]


def embed_texts(
    texts: List[str],
    model: Optional[Any] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> List[List[float]]:
    if not texts:
        return []

    model_instance = model or load_embedding_model(model_name=model_name)
    vectors = model_instance.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_functions(
    functions: List[Dict[str, object]],
    model: Optional[Any] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> List[Dict[str, object]]:
    texts = build_function_documents(functions)
    vectors = embed_texts(
        texts,
        model=model,
        model_name=model_name,
        batch_size=batch_size,
    )

    embedded_items = []
    for func, text, vector in zip(functions, texts, vectors):
        embedded_items.append(
            {
                "name": func.get("name"),
                "file": func.get("file"),
                "language": func.get("language"),
                "start_line": func.get("start_line"),
                "end_line": func.get("end_line"),
                "text": text,
                "embedding": vector,
            }
        )

    return embedded_items


def extract_and_embed_functions(
    repo_path: Optional[str] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model: Optional[Any] = None,
    batch_size: int = 32,
) -> Dict[str, object]:
    function_extractor = load_function_extractor_module()
    functions, scanned_file_count, filtered_file_count = function_extractor.extract_functions_from_filtered_files(
        repo_path=repo_path
    )
    model_instance = model or load_embedding_model(model_name=model_name)
    embedded_items = embed_functions(
        functions,
        model=model_instance,
        model_name=model_name,
        batch_size=batch_size,
    )

    embedding_dimension = len(embedded_items[0]["embedding"]) if embedded_items else 0

    return {
        "model_name": model_name,
        "filtered_file_count": filtered_file_count,
        "scanned_file_count": scanned_file_count,
        "function_count": len(functions),
        "embedding_dimension": embedding_dimension,
        "items": embedded_items,
    }


def score_functions_by_query(
    query: str,
    embedded_items: List[Dict[str, object]],
    model: Optional[Any] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> List[Dict[str, object]]:
    if not embedded_items:
        return []

    model_instance = model or load_embedding_model(model_name=model_name)
    query_vector = model_instance.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    scored_items: List[Dict[str, object]] = []
    for item in embedded_items:
        item_vector = np.asarray(item["embedding"], dtype=float)
        score = float(np.dot(query_vector, item_vector))

        scored_item = dict(item)
        scored_item["score"] = score
        scored_items.append(scored_item)

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items


def query_top_functions(
    query: str,
    top_k: int = 3,
    repo_path: Optional[str] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> Dict[str, object]:
    model = load_embedding_model(model_name=model_name)
    extraction_result = extract_and_embed_functions(
        repo_path=repo_path,
        model_name=model_name,
        model=model,
        batch_size=batch_size,
    )
    scored_items = score_functions_by_query(
        query=query,
        embedded_items=extraction_result["items"],
        model=model,
        model_name=model_name,
    )
    top_results = scored_items[:top_k]

    return {
        "query": query,
        "model_name": model_name,
        "filtered_file_count": extraction_result["filtered_file_count"],
        "scanned_file_count": extraction_result["scanned_file_count"],
        "function_count": extraction_result["function_count"],
        "embedding_dimension": extraction_result["embedding_dimension"],
        "top_k": top_k,
        "results": top_results,
    }


if __name__ == "__main__":
    result = extract_and_embed_functions()
    print(f"Model: {result['model_name']}")
    print(f"Filtered files: {result['filtered_file_count']}")
    print(f"Scanned files: {result['scanned_file_count']}")
    print(f"Functions: {result['function_count']}")
    print(f"Embedding dimension: {result['embedding_dimension']}")
    if result["items"]:
        print("Sample function:", result["items"][0]["name"])
        print("Sample file:", result["items"][0]["file"])