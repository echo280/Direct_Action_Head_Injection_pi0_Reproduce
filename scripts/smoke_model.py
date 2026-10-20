"""Remote JAX acceptance test. Uses random dummy experts, not downloaded weights."""
import dataclasses
import json
import numpy as np
from common import RUNS, setup

setup()
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models.pi0_config import Pi0Config
from openpi.spatial.mlp import SpatialMLP

mlp = SpatialMLP(64, rngs=nnx.Rngs(42))
d = jnp.array([[0.1, -0.2, 0.3]], dtype=jnp.float32)
np.testing.assert_array_equal(np.asarray(mlp(d)), np.zeros((1, 64)))
loss, gradients = nnx.value_and_grad(lambda net: jnp.mean((net(d) - 1.0) ** 2))(mlp)
assert np.linalg.norm(np.asarray(gradients["output"]["kernel"].value)) > 0
np.testing.assert_array_equal(np.asarray(gradients["input"]["kernel"].value), np.zeros((3, 1024)))

base_config = Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                       action_horizon=2, action_dim=8, max_token_len=4, dtype="float32")
spatial_config = dataclasses.replace(base_config, spatial_conditioning=True)
base = base_config.create(jax.random.key(123))
spatial = spatial_config.create(jax.random.key(123))
obs = base_config.fake_obs()
spatial_obs = obs.replace(spatial_displacement=d)
actions = base_config.fake_act()
t = jnp.array([0.5])
base_condition = base.embed_suffix(obs, actions, t)[-1]
spatial_condition = spatial.embed_suffix(spatial_obs, actions, t)[-1]
np.testing.assert_allclose(base_condition, spatial_condition, atol=1e-6, rtol=1e-6)
base_loss = base.compute_loss(jax.random.key(7), obs, actions, train=False)
spatial_loss = spatial.compute_loss(jax.random.key(7), spatial_obs, actions, train=False)
np.testing.assert_allclose(base_loss, spatial_loss, atol=1e-5, rtol=1e-5)
try:
    spatial.embed_suffix(obs, actions, t)
except ValueError:
    pass
else:
    raise AssertionError("Missing spatial signal must fail")
report = {"status": "passed", "jax_devices": [str(device) for device in jax.devices()],
          "checks": ["zero MLP output", "nonzero output-layer gradient", "initial zero first-layer gradient",
                     "matching original time condition", "matching model loss", "missing geometry rejected"],
          "scope": "random dummy experts; not pretrained-weight or rollout validation"}
(RUNS / "reports/model_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
