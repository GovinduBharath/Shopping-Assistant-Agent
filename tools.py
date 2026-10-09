import os
import requests

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")


def search_products(product, budget):

    url = "https://real-time-product-search.p.rapidapi.com/search"

    querystring = {
        "q": product,
        "country": "in",
        "language": "en",
        "page": "1",
        "limit": "10"
    }

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
    }

    response = requests.get(
        url,
        headers=headers,
        params=querystring,
        timeout=20
    )

    if response.status_code != 200:
        return []

    data = response.json()

    products = []

    for item in data.get("data", {}).get("products", []):

        products.append({
            "title": item.get("product_title"),
            "brand": item.get("brand"),
            "price": item.get("offer", {}).get("price"),
            "rating": item.get("product_rating"),
            "url": item.get("product_page_url"),
            "description": item.get("product_description")
        })

    return products
