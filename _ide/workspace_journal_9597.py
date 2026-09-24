# 2026-09-18T16:18:08.345280
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.get_component(name="FPED_ACO_AIE_code")
comp.build(target="x86sim")

comp.build(target="x86sim")

vitis.dispose()

