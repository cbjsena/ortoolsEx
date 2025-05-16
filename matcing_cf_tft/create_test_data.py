import json
import random
import os

num_cf_panels = 5
num_tft_panels = 5
panel_rows = 3
panel_cols = 3
defect_rate = 10

def create_panel_data(panel_id_prefix, num_panels, rows, cols, rate):
    panels = []
    for i_panel in range(1, num_panels + 1):
        defect_map = []
        for r_idx in range(rows):
            row_map = []
            for c_idx in range(cols):
                if random.randint(1, 100) <= rate:
                    row_map.append(1)
                else:
                    row_map.append(0)
            defect_map.append(row_map)
        panels.append({
            "id": f"{panel_id_prefix}{i_panel}",
            "rows": rows,
            "cols": cols,
            "defect_map": defect_map
        })
    return panels


generated_cf_panels = create_panel_data("CF", num_cf_panels, panel_rows, panel_cols, defect_rate)
generated_tft_panels = create_panel_data("TFT", num_tft_panels, panel_rows, panel_cols, defect_rate)

generated_data = {
    "panel_dimensions": {"rows": panel_rows, "cols": panel_cols},
    "cf_panels": generated_cf_panels,
    "tft_panels": generated_tft_panels,
    "settings": {
        "num_cf_panels": num_cf_panels,
        "num_tft_panels": num_tft_panels,
        "defect_rate_percent": defect_rate,
        "panel_rows": panel_rows,
        "panel_cols": panel_cols,
    }
}

base_dir = 'testcase'
os.makedirs(base_dir, exist_ok=True)

base_filename = f"test_cf{num_cf_panels}_tft{num_tft_panels}_row{panel_rows}_col{panel_cols}_rate{defect_rate}"
index = 1

while True:
    filename = f"{base_filename}_{index:03d}.json"
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        break
    index += 1

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(generated_data, f, indent=4, ensure_ascii=False)

print("✅ test.json 파일이 생성되었습니다.")
