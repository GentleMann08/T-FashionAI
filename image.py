# Пример базы данных товаров
product_database = {
    "tops": [
        {"id": "top1", "color": "red", "style": "casual"},
        {"id": "top2", "color": "blue", "style": "formal"},
    ],
    "bottoms": [
        {"id": "bottom1", "color": "red", "style": "casual"},
        {"id": "bottom2", "color": "blue", "style": "formal"},
    ],
    "shoes": [
        {"id": "shoe1", "color": "black", "style": "casual"},
        {"id": "shoe2", "color": "brown", "style": "formal"},
    ],
}

# Функция для создания комплектов
def create_outfit(style, color):
    outfit = {
        "top": None,
        "bottom": None,
        "shoes": None,
    }
    
    for category, items in product_database.items():
        for item in items:
            if item["style"] == style and (color is None or item["color"] == color):
                outfit[category] = item
                break
    
    return outfit

# Пример использования
user_style = "casual"
user_color_preference = "red"
outfit = create_outfit(user_style, user_color_preference)
print("Созданный образ:", outfit)
