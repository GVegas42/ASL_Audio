import mediapipe as mp
print('mediapipe version:', getattr(mp, '__version__', None))
print('has solutions attribute:', hasattr(mp, 'solutions'))
print('top-level attributes containing "solutions" or "hands":', [a for a in dir(mp) if 'solutions' in a or 'hands' in a])
try:
    import mediapipe.tasks.python as mp_tasks_py
    print('has mediapipe.tasks.python:', True)
    print('mediapipe.tasks.python attrs:', [a for a in dir(mp_tasks_py) if 'solutions' in a or 'hands' in a or 'vision' in a])
except Exception as e:
    print('cannot import mediapipe.tasks.python:', e)
