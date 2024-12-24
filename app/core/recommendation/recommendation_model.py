from redis import Redis
import pandas as pd
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from torch.nn.functional import cosine_similarity

from app.hepler.ultils import json_to_dataframe
from app.hepler.similarity import (
    compute_cosine_similarity,
    contains_word,
    extract_words,
    normalize_scores,
)
from app.schema.job import Job
from app.schema.recommendation import RecommendationRequest, PreRecommendationRequest, NewProductRecommendationRequest

model = SentenceTransformer("paraphrase-mpnet-base-v2")
img_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def pre_recommend(redis: Redis, data: RecommendationRequest):
    products = data.products
    scores = data.type_scores
    number_of_items = data.number_of_items

    df = json_to_dataframe(products)
    df_types = df.groupby("product_type").apply(
        lambda group: [
            (row["name"], row["shopify_id"])
            for _, row in group.iterrows()
        ]
    )

    types=df_types.keys()
    df_new = sort_for_similarity(types, scores)

    recommend_items = {}
    for item in df.itertuples():
        type = item.product_type
        id = item.shopify_id
        i=0
        recommend_items[id]=[]

        recommend_types = df_new[type]
        for recommend_type in recommend_types:
            chosen_product_index = random.randint(0, len(df_types[recommend_type])-1)
            chosen_product_id = df_types[recommend_type][chosen_product_index][1]
            if len(chosen_product_id)>-1:
                recommend_items[id].append(chosen_product_id)
                i+=1
            if i==number_of_items:
                break

    return recommend_items


def recommend(redis: Redis, data: dict):
    request_data = RecommendationRequest(**data)
    products = request_data.products
    type_scores = request_data.type_scores
    number_of_items = request_data.number_of_items
    product_score = request_data.product_scores

    df = json_to_dataframe(products)
    df["name_embedding"] = None
    df["image_embedding"] = None
    df["description_embedding"] = None
    results = []

    for row in df.itertuples():
        try:
            name_embedding = None
            try:
                name_embedding = model.encode(row.name)
            except Exception as e:
                print(row.name)
                print(f"Lỗi với sản phẩm {row.name}: {e}")

            desciption_embedding = None
            try:
                desciption_embedding = model.encode(row.description)
            except Exception as e:
                print(row.description)
                print(f"Lỗi với sản phẩm {row.name}: {e}")

            image_embedding = None
            try:
                response = requests.get(row.image)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    image_embedding = img_model.get_image_features(**inputs)
                    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                    image_embedding = image_embedding.reshape((512,))
            except Exception as e:
                print(f"Image")
                print(f"Lỗi với sản phẩm {row.name}: {e}")

            results.append({
                "index": row.Index,
                "name_embedding": name_embedding,
                "description_embedding": desciption_embedding,
                "image_embedding": image_embedding,
            })

        except Exception as e:
            print(f"Lỗi với sản phẩm {row.name}: {e}")

    for result in results:
        df.at[result["index"], "name_embedding"] = result["name_embedding"]
        df.at[result["index"], "image_embedding"] = result["image_embedding"]
        df.at[result["index"], "description_embedding"] = result["description_embedding"]


    df_types = df.groupby("product_type").apply(
        lambda group: [
            (row["shopify_id"], row["name"], row["name_embedding"], row["image_embedding"], row["description_embedding"])
            for _, row in group.iterrows()
        ]
    )

    types=df_types.keys()
    df_new = sort_for_similarity(types, type_scores)

    recommend_items = {}
    for item in df.itertuples():
        type = item.product_type
        id = item.shopify_id
        name_embedding = item.name_embedding
        image_embedding = item.image_embedding
        desciption_embedding = item.description_embedding

        if type not in df_new :
            continue

        recommend_types = df_new[type]
        recommend_items[id] = []
        i=0

        for recommend_type in recommend_types:
            max_similarity = -1
            chosen_product_id = ""
            for product_id, product_name, embedding_product_name, embedding_img, embedding_description in df_types[recommend_type]:
                try:
                    similarities = []             
                    try:
                        similarity1 = cosine_similarity_vector(
                            np.array(name_embedding),
                            np.array(embedding_product_name)
                        )
                        similarities.append(similarity1)
                    except Exception as e:
                        print(f"Lỗi tính similarity1 cho sản phẩm {product_name}: {e}")
                    
                    try:
                        similarity2 = cosine_similarity_vector(
                            image_embedding.numpy().reshape((512,)),
                            embedding_img.numpy().reshape((512,))
                        )
                        similarities.append(similarity2)
                    except Exception as e:
                        print(f"Lỗi tính similarity2 cho sản phẩm {product_name}: {e}")
                    
                    try:
                        similarity3 = cosine_similarity_vector(
                            np.array(desciption_embedding),
                            np.array(embedding_description)
                        )
                        similarities.append(similarity3)
                    except Exception as e:
                        print(f"Lỗi tính similarity3 cho sản phẩm {product_name}: {e}")

                    if similarities:  
                        similarity = sum(similarities) / len(similarities)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            chosen_product_id = product_id
                    else:
                        similarity = 0

                    score = product_score.get(id, {}).get(product_id, 0)

                    similarity = similarity + 3*score

                except Exception as e:
                    print(f"Lỗi tổng quát với sản phẩm {product_name}: {e}")

            if len(chosen_product_id)>0:
                recommend_items[id].append(chosen_product_id)
                i+=1

            if i==number_of_items:
                break

    return recommend_items

def new_product_recommend(redis: Redis, data: NewProductRecommendationRequest):
    product_id = data.product_id
    products = data.products
    number_of_items = data.number_of_items

    df = json_to_dataframe(products)
    df_types = df.groupby("product_type").apply(
        lambda group: [
            (row["name"], row["shopify_id"])
            for _, row in group.iterrows()
        ]
    )

    types=df_types.keys()
    embeddings = embedding_type(types)

    text = df.loc[df['shopify_id'] == product_id, 'product_type'].iloc[0]
    vector = embeddings[text]
    other_texts = {k: v for k, v in embeddings.items() if k != text}
    similarities = compute_cosine_similarity(vector, list(other_texts.values()))

    adjusted_similarities = []
    words_in_text = extract_words(text)

    for i, other_text in enumerate(other_texts.keys()):
        similarity = similarities[i]
        words_in_other_text = extract_words(other_text)

        filtered_words_in_text = words_in_text - {"men", "women", "s"}
        filtered_words_in_other_text = words_in_other_text - {"men", "women", "s"}

        common_words = filtered_words_in_text & filtered_words_in_other_text

        if common_words:
            similarity -= 0.5  
        adjusted_similarities.append(similarity)

    recommend_types = [t for t, _ in sorted(zip(other_texts.keys(), adjusted_similarities), key=lambda x: x[1], reverse=True)]
    
    recommend_list = []
    i=0
    for recommend_type in recommend_types:
        chosen_product_index = random.randint(0, len(df_types[recommend_type])-1)
        chosen_product_id = df_types[recommend_type][chosen_product_index][0]
        if len(chosen_product_id)>-1:
            recommend_list.append(chosen_product_id)
            i+=1
        if i==number_of_items:
            break
    
    return recommend_list

def embedding_type(texts):
    embeddings = {}
    for text in texts:
        text = str(text)
        embedding = model.encode(text)
        embeddings[text] = embedding
    return embeddings

def sort_for_similarity(texts, scores):
    embeddings = embedding_type(texts)

    df_new = {}
    for text, vector in embeddings.items():
        other_texts = {k: v for k, v in embeddings.items() if k != text}
        similarities = compute_cosine_similarity(vector, list(other_texts.values()))

        adjusted_similarities = []
        words_in_text = extract_words(text)

        for i, other_text in enumerate(other_texts.keys()):
            similarity = similarities[i]
            words_in_other_text = extract_words(other_text)

            filtered_words_in_text = words_in_text - {"men", "women", "s"}
            filtered_words_in_other_text = words_in_other_text - {"men", "women", "s"}

            common_words = filtered_words_in_text & filtered_words_in_other_text

            if common_words:
                similarity -= 0.5  
            score = normalize_scores(scores.get(text, {}).get(other_text, 0))
            adjusted_similarities.append(0.5*similarity+ 0.5*score)

        sorted_texts = [t for t, _ in sorted(zip(other_texts.keys(), adjusted_similarities), key=lambda x: x[1], reverse=True)]
        df_new[text] = sorted_texts
    return df_new

def cosine_similarity_vector(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)