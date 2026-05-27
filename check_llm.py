from backend.llm.emotion_tagger import tag_emotions
import inspect
src = inspect.getsource(tag_emotions)
print(src[:500])
