import os
import json
import csv

class DatasetValidator:
    def __init__(self, data_dir="data/raw"):
        self.invoices_dir = os.path.join(data_dir, "invoices")
        self.metadata_dir = os.path.join(data_dir, "metadata")
        self.manifest_path = os.path.join(self.metadata_dir, "dataset_manifest.json")
        self.catalog_path = os.path.join(self.metadata_dir, "dataset_catalog.csv")

    def validate_integrity(self):
        errors = []
        
        # 1. Check metadata files exist
        if not os.path.exists(self.manifest_path):
            errors.append("Manifest file missing")
            return False, errors
            
        if not os.path.exists(self.catalog_path):
            errors.append("Catalog CSV missing")
            return False, errors
            
        with open(self.manifest_path) as f:
            manifest = json.load(f)
            
        catalog = []
        with open(self.catalog_path) as f:
            dr = csv.DictReader(f)
            for row in dr:
                catalog.append(row)
                
        # 7. Manifest counts match actual records
        if manifest["total_invoice_count"] != len(catalog):
            errors.append("Manifest count mismatch")
            
        seen_ids = set()
        
        for row in catalog:
            inv_id = row["invoice_id"]
            
            # 3. Invoice IDs are unique
            if inv_id in seen_ids:
                errors.append(f"Duplicate invoice ID: {inv_id}")
            seen_ids.add(inv_id)
            
            img_path = os.path.join(self.invoices_dir, row["source_file"])
            json_path = os.path.join(self.invoices_dir, f"{inv_id}.json")
            
            # 1. Referenced files exist
            if not os.path.exists(img_path):
                errors.append(f"Missing image for {inv_id}")
            if not os.path.exists(json_path):
                errors.append(f"Missing JSON for {inv_id}")
                continue
                
            # 2. Ground-truth JSON loads correctly
            try:
                with open(json_path) as f:
                    gt = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load JSON {inv_id}: {e}")
                continue
                
            # 4. Required structural fields
            required_keys = ["invoice_id", "source_file", "seller", "buyer", "items", "totals", "ground_truth_metadata"]
            for k in required_keys:
                if k not in gt:
                    errors.append(f"{inv_id} missing structural field {k}")
                    
            # 5. Numeric fields are numeric
            if "totals" in gt:
                for k, v in gt["totals"].items():
                    if not isinstance(v, (int, float)):
                        errors.append(f"{inv_id} non-numeric total {k}")
            
            # 6. Items have expected structure
            if "items" in gt and isinstance(gt["items"], list):
                for item in gt["items"]:
                    if "description" not in item or "taxable_value" not in item:
                        errors.append(f"{inv_id} item missing fields")
            
            # 8. Catalog agrees with annotations
            if gt["source_file"] != row["source_file"]:
                errors.append(f"{inv_id} source file mismatch in catalog")
                
            # 9. Duplicate relationships are valid
            if gt["ground_truth_metadata"].get("is_duplicate"):
                dup_of = gt["ground_truth_metadata"].get("duplicate_of")
                if not dup_of or dup_of not in [c["invoice_id"] for c in catalog]:
                    errors.append(f"{inv_id} has invalid duplicate_of")
                    
        # 10. No orphan ground-truth records
        for f in os.listdir(self.invoices_dir):
            if f.endswith(".json"):
                fid = f.replace(".json", "")
                if fid not in seen_ids:
                    errors.append(f"Orphan JSON record found: {f}")
                    
        return len(errors) == 0, errors
