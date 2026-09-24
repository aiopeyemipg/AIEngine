# 2026-09-21T15:53:22.474724
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.get_component(name="ABC_AI_Engine")
comp.build(target="x86sim")

comp = client.get_component(name="my_fist_example_AIE")
comp.build(target="x86sim")

comp = client.get_component(name="GA_AI_Engine")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.create_aie_component(name="GA_AIE", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "empty_aie_component")

comp = client.get_component(name="GA_AIE")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

client.delete_component(name="GA_AIE")

client.delete_component(name="AIE_Simple_Example")

comp = client.get_component(name="AIE_Simple_Example")
comp.build(target="x86sim")

vitis.dispose()

