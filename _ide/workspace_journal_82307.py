# 2026-09-21T11:57:25.598109
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.get_component(name="FPED_ACO_AIE_code")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

vitis.dispose()

