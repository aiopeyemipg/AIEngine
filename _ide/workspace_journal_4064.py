# 2026-09-21T16:06:57.588635
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="GA_AIE", part = "xcvc1902-viva1596-3HP-e-S", template = "installed_aie_examples/simple")

comp = client.get_component(name="GA_AIE")
comp.build(target="x86sim")

client.delete_component(name="my_fist_example_AIE")

client.delete_component(name="my_fist_example_AIE")

comp = client.get_component(name="FPED_AIE_Simple")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.create_aie_component(name="template_window_class", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "installed_aie_examples/template_window_class")

comp = client.get_component(name="template_window_class")
comp.build(target="x86sim")

comp = client.get_component(name="ABC_AI_Engine")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.get_component(name="FPED_AIE_Simple")
comp.build(target="x86sim")

comp = client.get_component(name="GA_AIE")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp = client.get_component(name="PSO_AI_Engine")
comp.build(target="x86sim")

vitis.dispose()

