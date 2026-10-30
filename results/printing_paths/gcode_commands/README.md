# Representative deposition commands / 代表性沉积命令

The two saved examples are Euler Momentum closed and Direct Momentum closed for the same representative structure (seed 1).

两组文件为同一代表结构（seed 1）的欧拉化Momentum与直接Momentum路径。

- `outputs/<method>/<method>.gcode`: complete saved commands / 完整命令。
- `outputs/<method>/route_steps.csv`: ordered movement and extrusion inputs / 逐段路径及挤出量。
- `snippets/`: original short command excerpts / 原有命令节选。
- `config/research_demo.json`: original geometry and demonstration process settings / 原几何及示例工艺配置。

From the repository root / 在代码包根目录运行：

```powershell
python results/printing_paths/gcode_commands/export_gcode.py
```

Uses only the Python standard library. It exports commands from the saved movement table using the original export functions, preserving path order. Outputs go to `regenerated/` and are checked byte-for-byte against the saved examples.

仅使用Python标准库，采用原导出函数从已有逐段数据生成命令，不重新寻路。结果写入 `regenerated/`，并与保存的G-code逐字节核对。

These are single-layer command demonstrations, marked `RESEARCH_DEMO_NOT_MACHINE_READY` in the original files; printer-specific temperature and startup/shutdown settings are not supplied.

原文件标记为单层命令示例，未配置具体打印机温度及启停指令，不作为已完成实机打印的记录。
