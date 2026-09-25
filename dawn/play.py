from __future__ import annotations

import tyro

from dawn.config.play_args import PlayCfg
from dawn.utils.playback import run_playback


def main(cfg: PlayCfg) -> None:
    from dawn.rl.runners import DawnEvalRunner

    run_playback(cfg, DawnEvalRunner)


if __name__ == "__main__":
    main(tyro.cli(PlayCfg))
