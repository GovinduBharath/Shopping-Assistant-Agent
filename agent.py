
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from tools import search_products

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "Missing Gemini API key. Set GOOGLE_API_KEY in Render Environment."
    )

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI Shopping Assistant.
Recommend products based on the user's product category, budget,
preferred brand, required features, and intended use.

Use only the supplied product search results for product listings,
prices, ratings, and URLs. Never invent product details.
If no products are found, explain that clearly.
Compare the available options and explain why they may fit the user's needs."""
    ),
    (
        "human",
        """Find suitable products using these requirements:

Product: {product}
Budget: {budget}
Preferred brand: {brand}
Required features: {features}
Purpose: {purpose}

Product search results:
{product_results}

Recommend suitable available options."""
    ),
])

# Important: this is the LLM chain, not the outer shopping_chain.
recommendation_chain = prompt | llm | StrOutputParser()


def shopping_agent(inputs: dict) -> str:
    product = inputs.get("product", "").strip()

    if not product:
        return "Please enter a product to search for."

    budget = inputs.get("budget", "No budget specified")
    brand = inputs.get("brand", "Any")
    features = inputs.get("features", "No specific features")
    purpose = inputs.get("purpose", "General use")

    try:
        products = search_products(
            product=product,
            budget=budget,
        )
    except Exception as exc:
        print(f"Product search failed: {exc}")
        products = []

    if products:
        product_results = "\n\n".join(
            (
                f"Product: {item.get('title', 'Unknown')}\n"
                f"Brand: {item.get('brand') or 'Not specified'}\n"
                f"Price: {item.get('price') or 'Not available'}\n"
                f"Rating: {item.get('rating') or 'Not available'}\n"
                f"URL: {item.get('url') or 'Not available'}\n"
                f"Description: {item.get('description') or 'Not available'}"
            )
            for item in products
        )
    else:
        product_results = (
            "No products were returned by the product search API. "
            "Do not invent product listings."
        )

    return recommendation_chain.invoke({
        "product": product,
        "budget": budget,
        "brand": brand,
        "features": features,
        "purpose": purpose,
        "product_results": product_results,
    })


# LangServe exposes this runnable through /shopping-agent/invoke.
shopping_chain = RunnableLambda(shopping_agent)
