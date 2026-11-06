"""Alias of shared.video_io (the module moved there so other engines use the same code)."""
import sys

from shared import video_io as _shared_video_io

sys.modules[__name__] = _shared_video_io
