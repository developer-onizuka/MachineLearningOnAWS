import random
import pandas as pd
import uuid

def generate_commnets(rating, templates, count):
    comments = []
    for _ in range(count):
        comment_body = random.choice(templates)
        comment = [rating, str(uuid.uuid4()), comment_body]
        comments.append(comment)
    return comments

def main():
    templates_5 = [
        "Absolutely loved it, it exceeded my expectations!",
        "This product is fantastic, I couldn't be happier!",
        "Highly recommend it, truly impressed by the quality!",
        "Amazing experience, I'll definitely buy again!",
        "A perfect choice, better than I could have hoped for!"
    ]
    templates_4 = [
        "Really liked it, but there's room for improvement.",
        "Great product, I would purchace it again.",
        "Very satisfied, though I wish it had a little more.",
        "Overall, a great choice with minor drawbacks.",
        "Solid quality, please with the purchase."
    ]
    templates_3 = [
        "It is okay, but it didn't really stand out.",
        "Average experience, Could be better.",
        "Not bad, but not great either.",
        "Decent product, met expectations but didn't exceed them.",
        "Neutral feeling, gets the job done but lacks wow factor."
    ]
    templates_2 = [
        "Not very impressed, there were some issues.",
        "Below my expectations, not worth the price.",
        "Disappointed, wouldn't buy this again.",
        "Quality wasn't up to the mark, wouldn't recommend.",
        "A poor experience, expected something better."
    ]
    templates_1 = [
        "Terrible product, completely unacceptable.",
        "Hated it, an absolute waste of money.",
        "Worst experience ever, stay away!",
        "Not as described, extremely poor quality.",
        "Regret buying this, can't recommend at all."
    ]
    num_comments = 10

    comments_5 = generate_commnets(5, templates_5, num_comments)
    comments_4 = generate_commnets(4, templates_4, num_comments)
    comments_3 = generate_commnets(3, templates_3, num_comments)
    comments_2 = generate_commnets(2, templates_2, num_comments)
    comments_1 = generate_commnets(1, templates_1, num_comments)
    
    all_comments = comments_5 + comments_4 + comments_3 + comments_2 + comments_1

    df = pd.DataFrame(all_comments, columns=["star_rating", "review_id", "review_body"])

    df.to_parquet('amazon_reviews_2015_small.snappy.parquet', engine='pyarrow', index=False)
    print("Data has been saved to 'amazon_reviews_2015_small.snappy.parquet'")

if __name__ == "__main__":
    main()
