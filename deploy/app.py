"""
Gradio wrapper for HF Spaces (ZeroGPU).

Exposes the search pipeline both as a Gradio UI and via Gradio's REST API
(/gradio_api/call/search) which the React frontend consumes.
"""

import spaces  # must be imported before torch/CUDA usage (ZeroGPU shim)
import gradio as gr

from main import run_search  # loads indexes + models at boot


@spaces.GPU(duration=30)
def search(q: str):
    """Returns the search response as JSON-serializable dict."""
    if not q or not q.strip():
        return {"status": "error", "query": q, "results": []}
    results, _ = run_search(q)
    return {"status": "success", "query": q, "results": results}


demo = gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Query", placeholder="rust closures"),
    outputs=gr.JSON(label="Results"),
    title="Hybrid Search Engine",
    description="BM25 + dense retrieval, RRF fusion, cross-encoder rerank.",
    api_name="search",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
