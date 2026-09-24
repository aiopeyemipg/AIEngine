# 2026-09-12T12:50:26.914296
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="single_node_template", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "installed_aie_examples/single_node_template")

comp = client.get_component(name="single_node_template")
comp.build(target="x86sim")

client.delete_component(name="single_node_template")

comp = client.create_aie_component(name="FPED_ACO_AIE_code", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "installed_aie_examples/simple")

comp = client.get_component(name="FPED_ACO_AIE_code")
comp.build(target="x86sim")

vitis.dispose()

