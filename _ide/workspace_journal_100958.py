# 2026-09-21T13:04:48.818852
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="my_fist_example_AIE", part = "xcvc1902-viva1596-3HP-e-S", template = "installed_aie_examples/template_window_class")

comp = client.get_component(name="my_fist_example_AIE")
comp.build(target="x86sim")

comp = client.get_component(name="FPED_ACO_AIE_code")
comp.build(target="x86sim")

comp.build(target="x86sim")

client.delete_component(name="GA_AI_Engine")

client.delete_component(name="PSO_AI_Engine")

client.delete_component(name="ABC_AI_Engine")

client.delete_component(name="FPED_ACO_AIE_code")

comp = client.get_component(name="my_fist_example_AIE")
comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

status = comp.clean(target="x86sim")

comp.build(target="x86sim")

status = comp.clean(target="x86sim")

comp.build(target="x86sim")

comp = client.create_aie_component(name="AIE_Simple_Example", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "installed_aie_examples/simple")

comp = client.get_component(name="AIE_Simple_Example")
comp.build(target="x86sim")

vitis.dispose()

