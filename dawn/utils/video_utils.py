import os

import gymnasium as gym
from isaaclab.utils.dict import print_dict


def record_video(env: gym.Env, log_dir: str, video_length: int, video_interval: int) -> gym.Env:
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % video_interval == 0,
        "video_length": video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    return env


def record_video_play(env: gym.Env, log_dir: str, video_length: int) -> gym.Env:
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "play"),
        "step_trigger": lambda step: step == 0,
        "video_length": video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during play.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    return env
