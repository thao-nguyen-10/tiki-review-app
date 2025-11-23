import requests
import pandas as pd
import time
import random
import os
from datetime import datetime

# ------------------------------
# CONFIG
# ------------------------------
CSV_PATH = "updated_reviews.csv"
KEYWORDS = ["sữa rửa mặt", "kem dưỡng", "tẩy trang", "tẩy tế bào chết", "son", "phấn"]
MAX_PRODUCTS_PER_KEYWORD = 10  # crawl first 10 products per keyword (for demo)
MAX_REVIEWS_PER_PRODUCT = 20   # max reviews to crawl per product
PRODUCTS_PER_PAGE = 40

# ------------------------------
# LOAD EXISTING CSV
# ------------------------------
if os.path.exists(CSV_PATH):
    df_base = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded existing CSV with {len(df_base)} rows")
else:
    df_base = pd.DataFrame(columns=['product_id','seller_id','review_id','customer_id','rating','content','created_time'])
    print("⚠ No existing CSV found, starting with empty DataFrame")

# ------------------------------
# HEADERS
# ------------------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
    'x-guest-token': 'VGi2I1R4OUJP8XxtETN0sYvwhn3lLo7D'
}

# ------------------------------
# 1. SEARCH PRODUCTS BY KEYWORD
# ------------------------------
def search_products(keyword, max_products=MAX_PRODUCTS_PER_KEYWORD):
    products = []
    page = 1
    while len(products) < max_products:
        params = {
            "limit": PRODUCTS_PER_PAGE,
            "page": page,
            "q": keyword
        }
        try:
            r = requests.get("https://tiki.vn/api/v2/products", headers=headers, params=params)
            if r.status_code != 200:
                break
            data = r.json().get("data", [])
            if not data:
                break

            for item in data:
                spid = item.get("id")
                seller_id = str(item.get("seller_id", 1))
                products.append({
                    # "spid": spid,
                    "product_id": spid,
                    "seller_id": seller_id
                })
                if len(products) >= max_products:
                    break

            page += 1
            time.sleep(random.uniform(1, 2))

        except Exception as e:
            print(f"⚠ Error searching products for '{keyword}': {e}")
            break

    print(f"🔍 Found {len(products)} products for keyword '{keyword}'")
    return products

# ------------------------------
# 2. CRAWL REVIEWS FOR PRODUCTS
# ------------------------------
def crawl_reviews(products, max_reviews=MAX_REVIEWS_PER_PRODUCT):
    reviews = []
    count = 0

    for p in products:
        for page in range(1, 6):  # crawl first 5 pages
            params = {
                "limit": 5,
                "include": "comments,contribute_info,attribute_vote_summary",
                "sort": "score|desc,id|desc",
                "page": page,
                # "spid": p['spid'],
                "product_id": p['product_id'],
                "seller_id": p['seller_id']
            }

            try:
                r = requests.get("https://tiki.vn/api/v2/reviews", headers=headers, params=params)
                if r.status_code != 200:
                    continue

                for record in r.json().get("data", []):
                    created_by = record.get('created_by')
                    created_time = created_by.get('created_time') if isinstance(created_by, dict) else None

                    reviews.append({
                        # 'spid': p['spid'],
                        'product_id': p['product_id'],
                        'seller_id': p['seller_id'],
                        'review_id': record.get('id'),
                        'customer_id': record.get('customer_id'),
                        'rating': record.get('rating'),
                        'content': record.get('content'),
                        'created_time': created_time or datetime.now().isoformat()
                    })
                    count += 1

                    if count >= max_reviews:
                        print(f"🆕 Crawled {count} new reviews (max)")
                        return pd.DataFrame(reviews)

                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"⚠ Error crawling {p['product_id']}: {e}")
                continue

    print(f"🆕 Crawled total {count} new reviews")
    return pd.DataFrame(reviews)

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    all_products = []
    for kw in KEYWORDS:
        products = search_products(kw)
        all_products.extend(products)

    # Remove duplicates by product_id
    all_products_unique = {p['product_id']:p for p in all_products}.values()

    df_new = crawl_reviews(list(all_products_unique), MAX_REVIEWS_PER_PRODUCT)

    if not df_new.empty:
        df_all = pd.concat([df_base, df_new], ignore_index=True)
        df_all.to_csv(CSV_PATH, index=False)
        print(f"💾 Saved updated CSV with {len(df_all)} rows to {CSV_PATH}")
    else:
        print("⚠ No new reviews crawled.")