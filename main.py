from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import whisper
import os

app = FastAPI()

# Load the model
model = whisper.load_model("medium")

def transcribe_audio(audio_path: str) -> str:
    # Load and transcribe the audio file
    result = model.transcribe(audio_path)

    # Extracting the transcription text
    transcription = result["text"]
    
    return transcription

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    audio_file_path = f"data/{file.filename}"
    
    # Save the uploaded file
    with open(audio_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get the transcription
    text = transcribe_audio(audio_file_path)
    
    # Save text to txt
    transcription_file_path = f"data/{file.filename}.txt"
    with open(transcription_file_path, "w") as file:
        file.write(text)
    
    return {"transcription_file": transcription_file_path}

@app.get("/download-transcription/")
async def download_transcription(file_name: str = "transcription"):
    transcription_file_path = f"data/{file_name}.txt"
    if os.path.exists(transcription_file_path):
        return FileResponse(transcription_file_path, media_type='application/octet-stream', filename=f"{file_name}.txt")
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)