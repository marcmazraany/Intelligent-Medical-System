import json, pandas as pd
from pathlib import Path

# 1) Load one OpenFDA file (either a JSON Lines or a JSON with "results")
def load_openfda(path: str):
    text = Path(path).read_text(encoding='utf-8')
    obj = json.loads(text)
    results = obj.get('results', obj)  # if it’s already an obj with fields, adapt here
    if isinstance(results, dict):
        results = [results]
    return results

records = load_openfda("Resources/drug-label-0001-of-0013.json")  # your file

# 2) Normalize top-level + openfda block
rows = []
for r in records:
    of = r.get("openfda", {})
    rows.append({
        "effective_time": r.get("effective_time"),
        "version": r.get("version"),
        "brand_names": of.get("brand_name", []),
        "generic_names": of.get("generic_name", []),
        "manufacturer_names": of.get("manufacturer_name", []),
        "product_ndcs": of.get("product_ndc", []),
        "package_ndcs": of.get("package_ndc", []),
        "upcs": of.get("upc", []),
        "route": of.get("route", []),
        "product_type": of.get("product_type", []),
        "substance_names": of.get("substance_name", []),
        "uniis": of.get("unii", []),
        "pharm_class_epc": of.get("pharm_class_epc", []),
        "pharm_class_pe": of.get("pharm_class_pe", []),
        "pharm_class_cs": of.get("pharm_class_cs", []),
        "spl_id": of.get("spl_id", []),
        "spl_set_id": of.get("spl_set_id", []),
        "is_original_packager": of.get("is_original_packager", [])
    })

df = pd.DataFrame(rows)

# 3) Explode to “tidy” subtables
def explode_list(df, col):
    return df.explode(col).dropna(subset=[col]).rename(columns={col: col[:-1] if col.endswith('s') else col})

products = explode_list(df, "product_ndcs")
packages = explode_list(df, "package_ndcs")
brands   = explode_list(df, "brand_names")
generics = explode_list(df, "generic_names")
upcs     = explode_list(df, "upcs")
routes   = explode_list(df, "route")
pclasses_epc = explode_list(df, "pharm_class_epc")
pclasses_pe  = explode_list(df, "pharm_class_pe")
pclasses_cs  = explode_list(df, "pharm_class_cs")

# Ingredients table: match substance_names with uniis by index
ing_rows = []
for _, row in df.iterrows():
    subs = row["substance_names"]
    ids  = row["uniis"]
    for i, name in enumerate(subs or []):
        unii = ids[i] if ids and i < len(ids) else None
        ing_rows.append({
            "effective_time": row["effective_time"],
            "brand_names": row["brand_names"],
            "generic_names": row["generic_names"],
            "substance_name": name,
            "unii": unii
        })

ingredients = pd.DataFrame(ing_rows)

# Create simplified CSV files with only brand names and generic names
# Extract unique brand names
all_brand_names = []
for brand_list in df["brand_names"]:
    if brand_list:
        all_brand_names.extend(brand_list)
unique_brands = pd.DataFrame({"brand_name": sorted(set(all_brand_names))})

# Extract unique generic names
all_generic_names = []
for generic_list in df["generic_names"]:
    if generic_list:
        all_generic_names.extend(generic_list)
unique_generics = pd.DataFrame({"generic_name": sorted(set(all_generic_names))})

# Save to CSV
unique_brands.to_csv("brand_names.csv", index=False)
unique_generics.to_csv("generic_names.csv", index=False)

print(f"Saved {len(unique_brands)} unique brand names to brand_names.csv")
print(f"Saved {len(unique_generics)} unique generic names to generic_names.csv")
