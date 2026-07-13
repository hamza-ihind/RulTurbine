import os
import urllib.request
import zipfile

def download_and_extract():
    # Target directories
    raw_dir = os.path.abspath("data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, "CMAPSSData.zip")
    
    # We will try a few mirror links for CMAPSSData.zip
    urls = [
        "https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1", # Zenodo mirror
        "https://raw.githubusercontent.com/MEK-0/Turbofan-RUL-Prediction-LSTM/master/CMAPSSData.zip", # GitHub raw zip if available
        "https://github.com/mapr-demos/predictive-maintenance/raw/master/originalDataSet/CMAPSSData.zip" # GitHub raw zip 2
    ]
    
    downloaded = False
    for url in urls:
        print(f"Trying to download from {url}...")
        try:
            # Set a user-agent to avoid HTTP 403 Forbidden
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully downloaded to {zip_path}")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {url}. Error: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
    if not downloaded:
        raise RuntimeError("Failed to download C-MAPSS dataset from all available sources.")
        
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract only the FD001 files we need
        fd001_files = [f for f in zip_ref.namelist() if "FD001" in f]
        for f in fd001_files:
            zip_ref.extract(f, raw_dir)
            print(f"Extracted: {f}")
            
    print("Download and extraction complete!")

if __name__ == "__main__":
    download_and_extract()
