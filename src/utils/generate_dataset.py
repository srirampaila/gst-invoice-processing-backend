import os
import json
import csv
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter

INVOICES_DIR = "data/raw/invoices"
METADATA_DIR = "data/raw/metadata"
os.makedirs(INVOICES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def create_invoice_image(filename, text, rotation=0, noisy=False):
    img = Image.new('RGB', (800, 1000), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Just draw text for synthetic invoice representation
    y_text = 50
    for line in text.split('\n'):
        d.text((50, y_text), line, fill=(0, 0, 0))
        y_text += 20
    
    if rotation:
        img = img.rotate(rotation, expand=True, fillcolor=(255, 255, 255))
    if noisy:
        img = img.filter(ImageFilter.GaussianBlur(1))
        
    img.save(os.path.join(INVOICES_DIR, filename))

def generate():
    invoices = []
    catalog = []
    
    # 5 CLEAN
    for i in range(1, 6):
        inv_id = f"INV_CLEAN_{i:03d}"
        invoices.append({
            "id": inv_id, "cat": "clean", "items": 1, "missing": [], "invalid": [], "rot": 0, "noisy": False
        })
    # 5 MULTI_ITEM
    for i in range(1, 6):
        inv_id = f"INV_MULTI_{i:03d}"
        invoices.append({
            "id": inv_id, "cat": "multi_item", "items": 3, "missing": [], "invalid": [], "rot": 0, "noisy": False
        })
    # 4 NOISY
    for i in range(1, 5):
        inv_id = f"INV_NOISY_{i:03d}"
        invoices.append({
            "id": inv_id, "cat": "noisy", "items": 1, "missing": [], "invalid": [], "rot": 0, "noisy": True
        })
    # 3 ROTATED
    for i in range(1, 4):
        inv_id = f"INV_ROT_{i:03d}"
        invoices.append({
            "id": inv_id, "cat": "rotated", "items": 1, "missing": [], "invalid": [], "rot": 5 * i, "noisy": False
        })
    # 3 MISSING_FIELDS
    for i in range(1, 4):
        inv_id = f"INV_MISS_{i:03d}"
        invoices.append({
            "id": inv_id, "cat": "missing_fields", "items": 1, "missing": ["buyer_gstin"], "invalid": [], "rot": 0, "noisy": False
        })
    # 3 INVALID
    for i in range(1, 4):
        inv_id = f"INV_INV_{i:03d}"
        invoices.append({
            "id": inv_id, "cat": "invalid", "items": 1, "missing": [], "invalid": ["math"], "rot": 0, "noisy": False
        })
        
    # Generate files
    for inv in invoices:
        filename = f"{inv['id']}.png"
        text = f"INVOICE NUMBER: {inv['id']}\nDATE: 2026-08-15\nSELLER GSTIN: 29ABCDE1234F1Z5\n"
        
        b_gstin = "27XYZAB5678C1Z2" if "buyer_gstin" not in inv['missing'] else None
        if b_gstin:
            text += f"BUYER GSTIN: {b_gstin}\n"
        
        items = []
        tv_total = 0
        for j in range(inv['items']):
            tv = 1000.0 * (j+1)
            tv_total += tv
            items.append({
                "description": f"Product {j}",
                "quantity": 1,
                "unit_price": tv,
                "taxable_value": tv,
                "gst_rate": 18.0,
                "cgst": tv * 0.09,
                "sgst": tv * 0.09,
                "igst": 0.0,
                "cess": 0.0
            })
            text += f"ITEM {j}: {tv} (Taxable), CGST: {tv*0.09}, SGST: {tv*0.09}\n"
            
        cgst_total = tv_total * 0.09
        sgst_total = tv_total * 0.09
        
        gt = tv_total + cgst_total + sgst_total
        if "math" in inv["invalid"]:
            gt += 500  # Intentional error
            
        text += f"TOTAL TAXABLE: {tv_total}\nCGST: {cgst_total}\nSGST: {sgst_total}\nGRAND TOTAL: {gt}\n"
        
        create_invoice_image(filename, text, rotation=inv['rot'], noisy=inv['noisy'])
        
        gt_data = {
            "invoice_id": inv['id'],
            "source_file": filename,
            "invoice_number": inv['id'],
            "invoice_date": "2026-08-15",
            "seller": {"name": "Seller", "gstin": "29ABCDE1234F1Z5", "address": "Blr"},
            "buyer": {"name": "Buyer", "gstin": b_gstin, "address": "Mum"},
            "place_of_supply": "27",
            "items": items,
            "totals": {
                "taxable_value": tv_total,
                "cgst": cgst_total,
                "sgst": sgst_total,
                "igst": 0.0,
                "cess": 0.0,
                "grand_total": gt
            },
            "ground_truth_metadata": {
                "quality": "noisy" if inv['noisy'] else ("rotated" if inv['rot'] else "clean"),
                "expected_review": bool(inv['missing'] or inv['invalid']),
                "is_duplicate": False,
                "duplicate_of": None,
                "missing_fields": inv['missing'],
                "intentional_errors": inv['invalid']
            }
        }
        
        with open(os.path.join(INVOICES_DIR, f"{inv['id']}.json"), 'w') as f:
            json.dump(gt_data, f, indent=2)
            
        catalog.append({
            "invoice_id": inv['id'],
            "source_file": filename,
            "category": inv['cat'],
            "file_format": "png",
            "item_count": inv['items'],
            "has_cgst": True,
            "has_sgst": True,
            "has_igst": False,
            "has_cess": False,
            "missing_fields": len(inv['missing']) > 0,
            "is_invalid": len(inv['invalid']) > 0,
            "is_duplicate": False,
            "duplicate_of": "",
            "expected_review": gt_data["ground_truth_metadata"]["expected_review"]
        })
        
    # DUPLICATE (2)
    for i in range(1, 3):
        orig_id = invoices[i]['id']
        inv_id = f"INV_DUP_{i:03d}"
        filename = f"{inv_id}.png"
        
        # Copy image
        with Image.open(os.path.join(INVOICES_DIR, f"{orig_id}.png")) as src_img:
            src_img.save(os.path.join(INVOICES_DIR, filename))
            
        # Read orig json
        with open(os.path.join(INVOICES_DIR, f"{orig_id}.json"), 'r') as f:
            gt_data = json.load(f)
            
        gt_data["invoice_id"] = inv_id
        gt_data["source_file"] = filename
        gt_data["ground_truth_metadata"]["is_duplicate"] = True
        gt_data["ground_truth_metadata"]["duplicate_of"] = orig_id
        
        with open(os.path.join(INVOICES_DIR, f"{inv_id}.json"), 'w') as f:
            json.dump(gt_data, f, indent=2)
            
        catalog.append({
            "invoice_id": inv_id,
            "source_file": filename,
            "category": "duplicate",
            "file_format": "png",
            "item_count": invoices[i]['items'],
            "has_cgst": True,
            "has_sgst": True,
            "has_igst": False,
            "has_cess": False,
            "missing_fields": False,
            "is_invalid": False,
            "is_duplicate": True,
            "duplicate_of": orig_id,
            "expected_review": False
        })
        
    # Write catalog
    cat_keys = catalog[0].keys()
    with open(os.path.join(METADATA_DIR, "dataset_catalog.csv"), 'w', newline='') as f:
        dw = csv.DictWriter(f, fieldnames=cat_keys)
        dw.writeheader()
        dw.writerows(catalog)
        
    # Write manifest
    manifest = {
        "dataset_version": "1.0",
        "creation_date": datetime.utcnow().isoformat(),
        "total_invoice_count": len(catalog),
        "category_counts": {
            "clean": 5, "multi_item": 5, "noisy": 4, "rotated": 3, "missing_fields": 3, "invalid": 3, "duplicate": 2
        },
        "file_formats": ["png"],
        "source_type": "synthetic",
        "synthetic": True,
        "duplicate_cases": 2,
        "missing_field_cases": 3,
        "invalid_cases": 3
    }
    with open(os.path.join(METADATA_DIR, "dataset_manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    generate()
