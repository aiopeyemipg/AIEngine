# 2026-09-12T11:55:00.142002
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="ACO_AIE", platform = "/tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm", template = "installed_aie_examples/simple")

comp = client.get_component(name="ACO_AIE")
comp.build(target="x86sim")

vitis.dispose()

