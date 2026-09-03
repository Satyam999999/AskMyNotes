import concurrent.futures
import requests
import time

url = "https://askmynotes-7gowy6fkla-uc.a.run.app/ask"
payload = {"question": "What are the core concepts explained in the text?"}
headers = {"Content-Type": "application/json"}

def hit_ask(req_id):
    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload)
        t = int((time.time() - t0)*1000)
        return resp.status_code, t
    except Exception as e:
        return 500, 0

def run_concurrent(n):
    print(f"🚀 Blasting {n} simultaneous requests to /ask...")
    t_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(hit_ask, i) for i in range(n)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    success = sum(1 for r in results if r[0] == 200)
    failed = len(results) - success
    times = [r[1] for r in results if r[0] == 200]
    avg_time = sum(times)/len(times) if times else 0
    max_time = max(times) if times else 0
    
    print(f"✅ Success: {success}/{n} | ❌ Failed (503s): {failed}/{n}")
    print(f"⏱️  Avg Latency: {avg_time:.0f}ms | Max Latency: {max_time}ms")
    print(f"Total time to resolve all requests: {int((time.time() - t_start)*1000)}ms\n")

print("--- WARMING UP ---")
requests.post(url, headers=headers, json=payload)

run_concurrent(5)
run_concurrent(10)
