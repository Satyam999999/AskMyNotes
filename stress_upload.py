import requests
import time
from reportlab.pdfgen import canvas

url = "https://askmynotes-7gowy6fkla-uc.a.run.app/upload"

def create_large_pdf(filename, num_pages):
    print(f"Generating synthetic {num_pages}-page PDF ({filename})...")
    c = canvas.Canvas(filename)
    for i in range(num_pages):
        c.drawString(100, 750, f"This is page {i+1} of {num_pages} for stress testing.")
        for j in range(40):
            c.drawString(100, 700 - (j * 15), f"Dummy text line {j} on page {i+1} to simulate dense academic text and test the PyMuPDF processing memory footprint.")
        c.showPage()
    c.save()

def test_upload(filename):
    print(f"\n🚀 Uploading {filename} to Cloud Run...")
    t0 = time.time()
    try:
        with open(filename, "rb") as f:
            resp = requests.post(url, files={"pdf": (filename, f, "application/pdf")})
        
        t = int((time.time() - t0))
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"✅ SUCCESS! Processed in {t} seconds.")
            print(f"Response: {resp.text}")
        else:
            print(f"❌ FAILED! Response: {resp.text}")
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT! Cloud Run killed the request (took > 300s).")
    except Exception as e:
        print(f"❌ ERROR: {e}")

# Generate PDFs
create_large_pdf("stress_50.pdf", 50)
create_large_pdf("stress_100.pdf", 100)

# Run Upload Tests
test_upload("stress_50.pdf")
test_upload("stress_100.pdf")
