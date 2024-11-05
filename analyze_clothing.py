from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_embedding(image_path):
    # Извлекаем эмбеддинг изображения
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding.squeeze().numpy()

# Пример базы данных эмбеддингов товаров
database_embeddings = {
    # "item1": get_embedding("path/to/item1.jpg"),
    # "item2": get_embedding("path/to/item2.jpg"),
}

def find_similar_items(query_image_path, database_embeddings, top_n=5):
    query_embedding = get_embedding(query_image_path)
    
    similarities = {}
    for item_name, embedding in database_embeddings.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities[item_name] = similarity

    # Сортируем по степени сходства
    similar_items = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
    return similar_items
