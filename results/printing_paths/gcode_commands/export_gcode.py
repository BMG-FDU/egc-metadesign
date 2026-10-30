from pathlib import Path
import csv
import json

def format_xy(value):
    return f"{value:.6f}"

def create_gcode(slug, steps, source, config, out_path):
    proc = config["process"]
    geom = config["geometry"]
    start = steps[0]["start_xy_mm"]
    lines = [
        "; RESEARCH_DEMO_NOT_MACHINE_READY -- command-generation and parser-validation example only",
        f"; source_method={source['definition']['method']} representative_seed={source['route']['seed']}",
        "; Source step order and DEPOSIT/TRAVEL flags are copied from the current operation trace without resequencing.",
        f"; XY transform: translation only dX={geom['translation_xy_mm'][0]:.12f} mm dY={geom['translation_xy_mm'][1]:.12f} mm; no scale/rotation/mirror.",
        f"; Single centerline layer: Z={proc['z_mm']:.3f} mm, layer_height={proc['layer_height_mm']:.3f} mm, line_width={proc['line_width_mm']:.3f} mm.",
        f"; Demo filament_diameter={proc['filament_diameter_mm']:.3f} mm, multiplier={proc['extrusion_multiplier']:.3f}, print_speed={proc['print_speed_mm_s']:.1f} mm/s, travel_speed={proc['travel_speed_mm_s']:.1f} mm/s.",
        "; Extrusion model: V[mm^3]=deposited_length[mm]*line_width[mm]*layer_height[mm].",
        "; Relative filament E increment[mm]=V/(pi*(filament_diameter/2)^2)*extrusion_multiplier.",
        "; G21=mm; G90=absolute XYZ; M83=relative E. Deposit steps carry positive E; travel steps have zero E increment.",
        "; Retraction is disabled for this one-layer demonstration. No material, temperature, homing, firmware macro, or machine startup/shutdown is specified.",
        "G21",
        "G90",
        "M83",
        f"G0 X{format_xy(start[0])} Y{format_xy(start[1])} Z{proc['z_mm']:.3f} F{proc['travel_speed_mm_s'] * 60:.0f} ; INITIAL_POSITIONING_NOT_IN_PATH_STATISTICS",
        "; BEGIN_SOURCE_ROUTE_MOVES",
    ]
    for step in steps:
        step["gcode_line_number"] = len(lines) + 1
        feed = proc["print_speed_mm_s"] * 60 if step["movement_type"] == "DEPOSIT" else proc["travel_speed_mm_s"] * 60
        words = f"G1 X{format_xy(step['end_xy_mm'][0])} Y{format_xy(step['end_xy_mm'][1])}"
        if step["movement_type"] == "DEPOSIT":
            words += f" E{step['e_delta_mm']:.8f}"
        words += f" F{feed:.0f}"
        edge_token = step["edge_id"] or "NONE"
        comment = (f"; source_step_id={step['source_step_id']} source_step={step['source_step']} "
                   f"edge_id={edge_token} category={step['source_category']} "
                   f"state={step['movement_type']} e_delta_mm={step['e_delta_mm']:.8f}")
        line = words + " " + comment
        step["gcode_command"] = line
        lines.append(line)
    lines.append("; END_SOURCE_ROUTE -- no machine shutdown or firmware end macro follows")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return lines

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    config = json.loads((root/'config'/'research_demo.json').read_text(encoding='utf-8'))
    for definition in config['source']['routes']:
        slug = definition['slug']
        with (root/'outputs'/slug/'route_steps.csv').open(encoding='utf-8-sig',newline='') as f:
            rows = list(csv.DictReader(f))
        steps = []
        for row in rows:
            steps.append({
                'start_xy_mm': [float(row['start_x_mm']),float(row['start_y_mm'])],
                'end_xy_mm': [float(row['end_x_mm']),float(row['end_y_mm'])],
                'movement_type': row['movement_type'],
                'e_delta_mm': float(row['e_delta_mm']),
                'edge_id': row['edge_id'],
                'source_step_id': row['source_step_id'],
                'source_step': row['source_step'],
                'source_category': row['source_category'],
            })
        source = {'definition':definition, 'route':{'seed':config['source']['representative_seed']}}
        output = root/'regenerated'/f'{slug}.gcode'
        create_gcode(slug,steps,source,config,output)
        original = root/'outputs'/slug/f'{slug}.gcode'
        assert output.read_bytes() == original.read_bytes(), f'Mismatch: {slug}'
        print(f'{slug}: {len(steps)} moves; regenerated commands match saved G-code exactly.')
