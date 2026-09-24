# 2026-09-18T17:17:16.041874
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="GA_AI_Engine", part = "xcvc1902-viva1596-3HP-e-S", template = "installed_aie_examples/simple")

comp = client.get_component(name="GA_AI_Engine")
comp.build(target="x86sim")

vitis.dispose()

