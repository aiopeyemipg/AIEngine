# 2026-07-15T12:47:57.916979
import vitis

client = vitis.create_client()
client.set_workspace(path="AI_engine")

comp = client.create_aie_component(name="AI_engine", part = "xave1752-nsvg1369-1LHJ-i-L", template = "empty_aie_component")

comp = client.get_component(name="AI_engine")
status = comp.import_files(from_loc="", files=["/home/embeddedai002/AI-Engine-main"])

status = comp.update_top_level_file(top_level_file="AI-Engine-main/laboratory/Host_app_Lab5/src/graph_kernel_stubs.cpp")

comp.build(target="x86sim")

comp.build(target="hw")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="hw")

comp.build(target="x86sim")

comp.build(target="x86sim")

comp.build(target="x86sim")

