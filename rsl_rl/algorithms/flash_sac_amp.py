from rsl_rl.algorithms.flash_sac import FlashSAC
from rsl_rl.modules.flash_sac_actor_critic import FlashSACActorCritic


class FlashSAC_AMP(FlashSAC):
    """FlashSAC with AMP support.

    AMP reward blending and discriminator updates are orchestrated by runners.
    """

    def __init__(self, actor_critic: FlashSACActorCritic, **kwargs):
        super().__init__(actor_critic, **kwargs)
