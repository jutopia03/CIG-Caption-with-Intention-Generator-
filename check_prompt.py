from backend.llm import emotion_tagger
import inspect
src = inspect.getsource(emotion_tagger)
print(src)
