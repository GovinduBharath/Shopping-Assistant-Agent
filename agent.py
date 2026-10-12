import os
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("AQ.Ab8RN6KQxphr6gKsK5cOD7K2oMAF7j-JTfk2KmsAab-scA1xrQ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI Shopping Assistant.

Your job is to help users find suitable products.

Analyze:
- Product category
- User requirements
- Budget
- Brand preference
- Important features
- Usage purpose

Compare available products.

For every recommended product include:
1. Product name
2. Brand
3. Price
4. Main features
5. Rating
6. Product URL

Do not invent products, prices, ratings, or URLs.

Keep the recommendations clear and useful.
"""
    ),
    (
        "human",
        """
Help me find products using these requirements:

Product: {product}
Budget: {budget}
Brand: {brand}
Important features: {features}
Purpose: {purpose}
"""
    )
])

chain = prompt | llm | StrOutputParser()


def shopping_agent(inputs):

    products = search_products(
        product=inputs["product"],
        budget=inputs["budget"]
    )

    product_text = "\n".join(
        [
            f"""
Product: {item.get('title')}
Brand: {item.get('brand')}
Price: {item.get('price')}
Rating: {item.get('rating')}
URL: {item.get('url')}
"""
            for item in products
        ]
    )

    result = chain.invoke({
        "product": inputs["product"],
        "budget": inputs["budget"],
        "brand": inputs["brand"],
        "features": inputs["features"],
        "purpose": inputs["purpose"]
    })

    return result + "\n\nPRODUCT SEARCH RESULTS:\n" + product_text


shopping_chain = chain
