import os
import csv
from src.preprocessing.pipeline import preprocess_image
from src.preprocessing.config import PreprocessingConfig

def main():
    invoices_dir = "data/raw/invoices"
    output_dir = "data/processed/final"
    os.makedirs(output_dir, exist_ok=True)
    
    config = PreprocessingConfig()
    
    results = []
    
    for filename in os.listdir(invoices_dir):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue
            
        input_path = os.path.join(invoices_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            res = preprocess_image(input_path, output_path, config)
            results.append({
                "filename": filename,
                "input_dimensions": f"{res['input_dimensions'][0]}x{res['input_dimensions'][1]}",
                "output_dimensions": f"{res['output_dimensions'][0]}x{res['output_dimensions'][1]}",
                "processing_time": res["processing_time"],
                "success": res["success"]
            })
        except Exception as e:
            results.append({
                "filename": filename,
                "input_dimensions": "N/A",
                "output_dimensions": "N/A",
                "processing_time": 0,
                "success": False,
                "error": str(e)
            })
            
    print(f"Processed {len(results)} images.")
    
if __name__ == "__main__":
    main()
