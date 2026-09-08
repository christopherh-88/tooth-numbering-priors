import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

def load_fdi_dataset(path, has_diagnosis):
    d = json.load(open(path))
    img_lookup = {im['id']: im for im in d['images']}
    q_lookup = {c['id']: c['name'] for c in d['categories_1']}
    p_lookup = {c['id']: c['name'] for c in d['categories_2']}
    diag_lookup = {c['id']: c['name'] for c in d['categories_3']} if has_diagnosis else None

    rows = []
    for a in d['annotations']:
        img = img_lookup[a['image_id']]
        x, y, w, h = a['bbox']
        cx = (x + w / 2) / img['width']
        cy = (y + h / 2) / img['height']
        quadrant = str(q_lookup[a['category_id_1']])
        position = str(p_lookup[a['category_id_2']])
        fdi = quadrant + position
        row = {
            'source_file': Path(path).name,
            'image_id': f"{Path(path).stem}_{a['image_id']}",
            'file_name': img['file_name'],
            'x_center': cx, 'y_center': cy,
            'width': w / img['width'], 'height': h / img['height'],
            'quadrant': quadrant, 'position': position, 'fdi': fdi,
        }
        if has_diagnosis:
            row['diagnosis'] = diag_lookup[a['category_id_3']]
        rows.append(row)
    return pd.DataFrame(rows)

df_enum = load_fdi_dataset(HERE / 'train_quadrant_enumeration.json', has_diagnosis=False)
df_disease_train = load_fdi_dataset(HERE / 'train_quadrant_enumeration_disease.json', has_diagnosis=True)
df_disease_val = load_fdi_dataset(HERE / 'validation_triple.json', has_diagnosis=True)

df_enum['split'] = 'train_quadrant_enumeration'
df_disease_train['split'] = 'train_quadrant_enumeration_disease'
df_disease_val['split'] = 'validation_triple'

full = pd.concat([df_enum, df_disease_train, df_disease_val], ignore_index=True)
full.to_csv(HERE / 'dentex_full_fdi_table.csv', index=False)
print('Total rows (teeth with full FDI code):', len(full))
print('By split:')
print(full.groupby('split').size())
print()
print('Unique images with full FDI code:', full['image_id'].nunique())
print()
print('FDI code value counts (top 10):')
print(full['fdi'].value_counts().head(10))
