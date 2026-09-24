# 2026-09-21T16:03:52.213086
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="FPED_AIE_Simple", part = "xcvc1902-viva1596-3HP-e-S", template = "installed_aie_examples/simple")

comp = client.get_component(name="FPED_AIE_Simple")
comp.build(target="x86sim")

comp = client.get_component(name="GA_AI_Engine")
comp.build(target="x86sim")

client.delete_component(name="GA_AI_Engine")

