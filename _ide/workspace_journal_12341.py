# 2026-09-18T16:52:42.497330
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="ABC_AI_Engine", part = "xcvc1902-viva1596-3HP-e-S", template = "installed_aie_examples/simple")

comp = client.get_component(name="ABC_AI_Engine")
comp.build(target="x86sim")

vitis.dispose()

