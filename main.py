from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import fitz
import tempfile
import subprocess
import re
import zipfile
from typing import Optional

# Environment configuration
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set!")

AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

app = FastAPI(title="TDS Assignment Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnswerResponse(BaseModel):
    answer: str

SYSTEM_PROMPT = (
    "You are a data science assignment solver. Follow these rules:\n"
    "1. For direct answers: Return ONLY the final answer without any text\n"
    "2. For calculations/file processing: Generate either:\n"
    "   - Python code wrapped in ```python\n```\n"
    "   - Bash commands wrapped in ```bash\n```\n"
    "3. The code/commands must:\n"
    "   - Read files from current directory if needed\n"
    "   - Print ONLY the answer as the last line\n"
    "4. Available tools: python, pandas, numpy, awk, grep, jq, etc.\n"
    "5. Never include explanations or comments in the output\n"
)

def extract_code_blocks(answer: str) -> dict:
    """Extract both python and bash code blocks from the answer"""
    return {
        'python': re.search(r'```python\n(.*?)\n```', answer, re.DOTALL),
        'bash': re.search(r'```bash\n(.*?)\n```', answer, re.DOTALL)
    }

def execute_code(code: str, lang: str, cwd: str) -> str:
    try:
        if lang == 'python':
            proc = subprocess.run(
                ['python', '-c', code],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
        elif lang == 'bash':
            proc = subprocess.run(
                ['bash', '-c', code],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
        else:
            return "Error: Unsupported language"
        
        if proc.returncode == 0:
            # Get the last line of output (assumes answer is printed last)
            output_lines = proc.stdout.strip().split('\n')
            return output_lines[-1] if output_lines else ""
        return f"Error: {proc.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"

async def handle_uploaded_file(file: UploadFile, temp_dir: str):
    if not file:
        return
    
    # Save main file
    file_path = os.path.join(temp_dir, file.filename)
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # Handle ZIP files
    if file.filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            pass

def get_llm_answer(question: str, context: str = "") -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    
    prompt = f"Question: {question[:2000]}\n"
    if context:
        prompt += f"Context:\n{context[:3000]}\n"
    prompt += "Respond with either:\n1. ONLY the answer\n2. Python code in ```python block\n3. Bash commands in ```bash block"
    
    payload = {
        "model": "gpt-4",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = httpx.post(AIPROXY_BASE_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return "Error: LLM service unavailable"
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/api/", response_model=AnswerResponse)
async def solve_question(
    question: str = Form(...),
    file: UploadFile = File(None)
):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Handle file upload
        if file:
            await handle_uploaded_file(file, temp_dir)
        
        # Get LLM response
        llm_response = get_llm_answer(question)
        
        # Check for code blocks
        code_blocks = extract_code_blocks(llm_response)
        
        # Execute Python if exists
        if code_blocks['python']:
            result = execute_code(code_blocks['python'].group(1), 'python', temp_dir)
            return AnswerResponse(answer=result)
        
        # Execute Bash if exists
        if code_blocks['bash']:
            result = execute_code(code_blocks['bash'].group(1), 'bash', temp_dir)
            return AnswerResponse(answer=result)
        
        # Return direct answer (first line only)
        return AnswerResponse(answer=llm_response.split('\n')[0].strip())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
