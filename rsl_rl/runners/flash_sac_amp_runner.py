"""FlashSAC + AMP training runner.

Reuses SACAMPRunner logic while enabling a dedicated runner class name
for configuration and registry selection.
"""

from rsl_rl.runners.sac_amp_runner import SACAMPRunner


class FlashSACAMPRunner(SACAMPRunner):
    pass
