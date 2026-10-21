"""Keep automated tests offline unless a test explicitly enables an Agent."""

import os


os.environ["ENABLE_EXERCISE_AGENT"] = "false"
os.environ["ENABLE_TUTOR_AGENT"] = "false"
os.environ["FORCE_LLM_EVERY_TURN"] = "false"
