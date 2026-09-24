# 2026-09-24T11:25:11.871364
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.get_component(name="GA_AIE")
comp.build(target="x86sim")

comp = client.get_component(name="FPED_AIE_Simple")
comp.build(target="x86sim")

comp = client.get_component(name="ABC_AI_Engine")
comp.build(target="x86sim")

comp = client.get_component(name="FPED_AIE_Simple")
comp.build(target="x86sim")

comp = client.get_component(name="GA_AIE")
comp.build(target="x86sim")

comp = client.get_component(name="PSO_AI_Engine")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

