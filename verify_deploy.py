"""Final deployment verification — tests all endpoints."""
import sys
try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

BASE = "http://127.0.0.1:8000"
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}: PASS {detail}")
    else:
        failed += 1
        print(f"  ❌ {name}: FAIL {detail}")

print("\n🔍 FINAL DEPLOYMENT CHECK\n" + "=" * 50)

# 1. Health
print("\n[1] Health Endpoint")
r = requests.get(f"{BASE}/health")
h = r.json()
check("GET /health", r.status_code == 200, f"→ {r.status_code}")
check("Status healthy", h.get("status") == "healthy")
check("Index loaded", h.get("index_loaded") is True)
check("Cache enabled", h.get("cache_enabled") is True)

# 2. Overview
print("\n[2] API Overview")
r = requests.get(f"{BASE}/api/overview")
o = r.json()
check("GET /api/overview", r.status_code == 200)
check("Document count > 0", o.get("document_count", 0) > 0, f"→ {o.get('document_count')}")
check("Category count = 20", o.get("category_count") == 20, f"→ {o.get('category_count')}")
check("Has sample queries", len(o.get("samples", [])) > 0, f"→ {len(o.get('samples', []))} samples")

# 3. Cluster Summary
print("\n[3] Cluster Summary")
r = requests.get(f"{BASE}/api/cluster-summary")
cs = r.json()
check("GET /api/cluster-summary", r.status_code == 200)
check("Has cluster data", cs.get("k", 0) > 0, f"→ k={cs.get('k')}")

# 4. Query
print("\n[4] Semantic Search Query")
r = requests.post(f"{BASE}/query", json={"query": "NASA space shuttle mission"})
q = r.json()
check("POST /query", r.status_code == 200)
check("Has result text", len(q.get("result", "")) > 50, f"→ {len(q.get('result', ''))} chars")
check("Has dominant_cluster", q.get("dominant_cluster") is not None, f"→ cluster {q.get('dominant_cluster')}")
check("Result format [rank] [cat] (sim=X)", "[1]" in q.get("result", "") and "sim=" in q.get("result", ""))

# 5. Cache
print("\n[5] Cache System")
r = requests.get(f"{BASE}/cache/stats")
s = r.json()
check("GET /cache/stats", r.status_code == 200)
check("Cache tracking entries", s.get("total_entries", 0) >= 0, f"→ {s.get('total_entries')} entries")

# 6. Static Assets
print("\n[6] Static Assets (Visualizations)")
r1 = requests.get(f"{BASE}/assets/tsne_clusters.png")
check("t-SNE image loads", r1.status_code == 200, f"→ {len(r1.content)//1024} KB")
r2 = requests.get(f"{BASE}/assets/bic_curve.png")
check("BIC curve image loads", r2.status_code == 200, f"→ {len(r2.content)//1024} KB")

# 7. Dashboard UI
print("\n[7] Dashboard UI")
r = requests.get(f"{BASE}/")
html = r.text
check("GET / dashboard", r.status_code == 200)
check("Google Fonts loaded", "fonts.googleapis.com" in html)
check("Inter font family", "Inter" in html)
check("Has favicon", "icon" in html.lower())
check("Has meta description", 'meta name="description"' in html)
check("Has structured results JS", "parseResults" in html)
check("Has loading spinner", "loading-spinner" in html)
check("Has tech badges footer", "tech-badge" in html)
check("Has t-SNE chart section", "tsne_clusters" in html)
check("Has BIC chart section", "bic_curve" in html)

# 8. Swagger Docs
print("\n[8] API Documentation")
r = requests.get(f"{BASE}/docs")
check("GET /docs (Swagger)", r.status_code == 200)

# 9. Error Handling
print("\n[9] Error Handling")
r = requests.post(f"{BASE}/query", json={"query": ""})
check("Empty query → 400", r.status_code == 400)

# 10. GZip
print("\n[10] Performance")
r = requests.get(f"{BASE}/api/overview", headers={"Accept-Encoding": "gzip"})
has_gzip = r.headers.get("content-encoding") == "gzip"
check("GZip compression active", has_gzip, f"→ content-encoding: {r.headers.get('content-encoding', 'none')}")

# 11. Security Headers
print("\n[11] Security Headers")
r = requests.get(f"{BASE}/health")
check("X-Content-Type-Options", r.headers.get("x-content-type-options") == "nosniff")
check("X-Frame-Options", r.headers.get("x-frame-options") == "DENY")
check("Strict-Transport-Security", "max-age" in r.headers.get("strict-transport-security", ""))
check("Content-Security-Policy", "default-src" in r.headers.get("content-security-policy", ""))

# Summary
print("\n" + "=" * 50)
total = passed + failed
print(f"📊 RESULT: {passed}/{total} checks passed")
if failed == 0:
    print("🚀 ALL CLEAR — Ready for deployment!")
else:
    print(f"⚠️  {failed} check(s) need attention")
print()
