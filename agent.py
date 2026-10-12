
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import search_products

load_dotenv()

# Gemini API configuration
GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    raise ValueError(
        "Gemini API key is missing. Set GOOGLE_API_KEY in Render Environment."
    )

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# Shopping assistant prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI Shopping Assistant.

Help users find suitable products based on their requirements.

Consider:
- Product category
- Budget
- Preferred brand
- Required features
- Purpose of use

Use the provided product search results to make recommendations.
Compare available prices and features where possible.

Never invent product listings, prices, ratings, or URLs.
If no products are found, explain that and suggest trying another search.
Clearly state when information is unavailable.
"""
    ),
    (
        "human",
        """
Find suitable products using these requirements:

Product: {product}
Budget: {budget}
Preferred brand: {brand}
Required features: {features}
Purpose: {purpose}

Product search results:
{product_results}

Recommend the most suitable available options.
Explain briefly why each product matches the user's needs.
"""
    ),
])

# Build the LangChain runnable
shopping_chain = (
    prompt
    | llm
    | StrOutputParser()
)


def shopping_agent(inputs):
    """Search for products and generate AI recommendations."""

    products = search_products(
        product=inputs["product"],
        budget=inputs["budget"],
    )

    product_results = "\n\n".join(
        [
            (
                f"Product: {item.get('title', 'Unknown')}\n"
                f"Brand: {item.get('brand') or 'Not specified'}\n"
                f"Price: {item.get('price') or 'Not available'}\n"
                f"Rating: {item.get('rating') or 'Not available'}\n"
                f"URL: {item.get('url') or 'Not available'}\n"
                f"Description: {item.get('description') or 'Not available'}"
            )
            for item in products
        ]
    )

    if not product_results:
        product_results = (
            "No products were returned by the product search API."
        )

    return shopping_chain.invoke({
        "product": inputs["product"],
        "budget": inputs["budget"],
        "brand": inputs.get("brand", "Any"),
        "features": inputs.get("features", "No specific features"),
        "purpose": inputs.get("purpose", "General use"),
        "product_results": product_results,
    })


# Expose the callable function as a LangServe-compatible runnable.
from langchain_core.runnables import RunnableLambda

shopping_chain = RunnableLambda(shopping_agent)
