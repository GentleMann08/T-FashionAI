from sklearn.neighbors import NearestNeighbors
import numpy as np

# Пример данных о предпочтениях пользователей
# (user_id: [стиль, цвет, бренд, цена и т. д.])
user_preferences = {
    "user1": [1, 2, 0, 3],
    "user2": [0, 2, 1, 1],
    "user3": [3, 1, 2, 2],
    # и так далее
}

# Преобразуем словарь в массив numpy
user_ids = list(user_preferences.keys())
preference_matrix = np.array(list(user_preferences.values()))

# Модель для поиска похожих пользователей
def get_similar_users(user_id, top_n=3):
    model = NearestNeighbors(n_neighbors=top_n, metric='cosine')
    model.fit(preference_matrix)
    
    user_index = user_ids.index(user_id)
    distances, indices = model.kneighbors([preference_matrix[user_index]])
    
    similar_users = [user_ids[idx] for idx in indices.flatten() if idx != user_index]
    return similar_users

def personalized_recommendation(user_id, product_database):
    similar_users = get_similar_users(user_id)
    recommended_items = []
    
    # Предлагаем товары, которые предпочитают похожие пользователи
    for sim_user in similar_users:
        recommended_items.extend(product_database.get(sim_user, []))
    
    return recommended_items
