# 2026-09-21T14:38:09.347013
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.get_component(name="AIE_Simple_Example")
status = comp.clean(target="x86sim")

comp.build(target="x86sim")

comp.build(target="hw")

comp = client.get_component(name="my_fist_example_AIE")
status = comp.clean(target="x86sim")

comp.build(target="x86sim")

status = comp.clean(target="x86sim")

comp.build(target="x86sim")

status = comp.clean(target="x86sim")

comp.build(target="x86sim")

comp = client.get_component(name="AIE_Simple_Example")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.create_aie_component(name="FPED_AIE_COMPONENT", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "empty_aie_component")

comp = client.get_component(name="FPED_AIE_COMPONENT")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.get_component(name="my_fist_example_AIE")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.get_component(name="PSO_AI_Engine")
comp.build(target="x86sim")

comp = client.get_component(name="GA_AI_Engine")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.get_component(name="ABC_AI_Engine")
comp.build(target="x86sim")

comp = client.get_component(name="FPED_ACO_AIE_code")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

client.delete_component(name="FPED_ACO_AIE_code")

comp = client.get_component(name="ABC_AI_Engine")
comp.build(target="x86sim")

comp.build(target="x86sim")

client.delete_component(name="FPED_AIE_COMPONENT")

client.delete_component(name="AIE_Simple_Example")

comp.build(target="x86sim")

vitis.dispose()

