from dawn.rl.networks.wm_actor_net import WMActorNetwork
from dawn.rl.networks.wm_critic_net import WMCriticNetwork
from dawn.rl.utils.registry import Registry

networks_registry = Registry()
networks_registry.add(WMActorNetwork)
networks_registry.add(WMCriticNetwork)
