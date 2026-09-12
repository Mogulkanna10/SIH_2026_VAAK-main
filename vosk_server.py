import asyncio
import websockets
import sys
import json
from vosk import Model, KaldiRecognizer, SpkModel

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    model_path = "model"

model = Model(model_path)
spk_model = SpkModel("spk_model")

async def recognize(websocket):
    print("Client connected")
    rec = None
    audio_received = False
    try:
        async for message in websocket:
            if isinstance(message, str):
                msg = json.loads(message)
                if 'config' in msg:
                    rec = KaldiRecognizer(model, msg['config']['sample_rate'])
                    rec.SetSpkModel(spk_model)
                elif 'eof' in msg:
                    if rec:
                        await websocket.send(rec.FinalResult())
                    break
            else:
                if rec:
                    if not audio_received:
                        await websocket.send(json.dumps({"ack": "first_audio_byte"}))
                        audio_received = True
                    if rec.AcceptWaveform(message):
                        await websocket.send(rec.Result())
                    else:
                        await websocket.send(rec.PartialResult())
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(recognize, "localhost", 2700):
        print("Vosk WebSocket server started on ws://localhost:2700")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

