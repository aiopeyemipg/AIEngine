# 2026-09-12T12:23:13.573220
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="simple", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "installed_aie_examples/simple")

comp = client.get_component(name="simple")
comp.build(target="x86sim")

comp = client.get_component(name="AI_engine")
comp.build(target="x86sim")

client.delete_component(name="ACO_AIE")

client.delete_component(name="AI_engine")

comp = client.get_component(name="simple")
comp.build(target="x86sim")

comp = client.create_aie_component(name="aie_component_ACO", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "empty_aie_component")

client.delete_component(name="simple")

comp = client.get_component(name="aie_component_ACO")
comp.build(target="x86sim")

comp.build(target="hw")

client.delete_component(name="aie_component_ACO")

vitis.dispose()

