@echo off
call C:\Users\manas\miniconda3\Scripts\activate.bat venv
python -c "import numpy, faiss, sentence_transformers, fastapi; print('ALL OK numpy', numpy.__version__)"
