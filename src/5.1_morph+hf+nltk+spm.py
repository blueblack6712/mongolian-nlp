def print_tokenizations():
    data = [
        {
            "line": 1,
            "original": "Сүүлийн таван жил дараалан төмөр замаар нүүрс аваагүй.",
            "tokenizations": {
                "Hugging Face BERT": ["▁Сүүлийн", "▁таван", "▁жил", "▁дараалан", "▁төмөр", "▁замаар", "▁нүүрс", "▁аваагүй", "."],
                "Custom Regex/NLTK": ["Сүүл", "ийн", "таван", "жил", "дараалан", "төмөр", "замаар", "нүүрс", "аваагүй", "."],
                "SentencePiece": ["▁Сүүлийн", "▁таван", "▁жил", "▁дараалан", "▁төмөр", "▁замаар", "▁нүүрс", "▁аваагүй", "."],
                "Morphological": ["Сүүл", "ийн", "таван", "жил", "дараалан", "төмөр", "замаар", "нүүрс", "аваагүй", "."]
            }
        },
        {
            "line": 2,
            "original": "Тэгэхээр урд хөрш нүүрсийг авах эсэхээс бүх зүйл шалтгаална.",
            "tokenizations": {
                "Hugging Face BERT": ["▁Тэгэхээр", "▁урд", "▁хөрш", "▁нүүрсийг", "▁авах", "▁эсэхээс", "▁бүх", "▁зүйл", "▁шалтгаална", "."],
                "Custom Regex/NLTK": ["Тэгэхээр", "ур", "д", "хөрш", "нүүрс", "ийг", "авах", "эсэхээс", "бүх", "зүйл", "шалтгаална", "."],
                "SentencePiece": ["▁Тэгэхээр", "▁урд", "▁хөрш", "▁нүүрсийг", "▁авах", "▁эсэхээс", "▁бүх", "▁зүйл", "▁шалтгаална", "."],
                "Morphological": ["Тэгэхээр", "ур", "д", "хөрш", "нүүрс", "ийг", "авах", "эсэх", "ээс", "бүх", "зүйл", "шалтгаална", "."]
            }
        },
        {
            "line": 3,
            "original": "Хятад руу төмөр замаар нүүрс тээвэрлэхэд бид гол анхаарлаа хандуулаад буй.",
            "tokenizations": {
                "Hugging Face BERT": ["▁Хятад", "▁руу", "▁төмөр", "▁замаар", "▁нүүрс", "▁тээвэрлэх", "эд", "▁бид", "▁гол", "▁анхаарлаа", "▁хандуул", "аад", "▁буй", "."],
                "Custom Regex/NLTK": ["Хята", "д", "", "руу", "төмөр", "замаар", "нүүрс", "тээвэрлэхэ", "д", "би", "д", "гол", "анхаарлаа", "хандуулаа", "д", "буй", "."],
                "SentencePiece": ["▁Хятад", "▁руу", "▁төмөр", "▁замаар", "▁нүүрс", "тээвэрлэх", "эд", "▁бид", "▁гол", "▁анхаарлаа", "▁хандуул", "аад", "▁буй", "."],
                "Morphological": ["Хята", "д", "руу", "төмөр", "замаар", "нүүрс", "тээвэрлэхэ", "д", "би", "д", "гол", "анхаарлаа", "хандуулаа", "д", "буй", "."]
            }
        },
        {
            "line": 4,
            "original": "Мөн өнгөрсөн жил сая тонн ачаа тээвэрлэсэн.",
            "tokenizations": {
                "Hugging Face BERT": ["▁Мөн", "▁өнгөрсөн", "▁жил", "▁сая", "▁тонн", "▁ачаа", "▁тээвэрлэсэн", "."],
                "Custom Regex/NLTK": ["Мөн", "өнгөрсөн", "жил", "сая", "тонн", "ачаа", "тээвэрлэсэн", "."],
                "SentencePiece": ["▁Мөн", "▁өнгөрсөн", "▁жил", "▁сая", "▁тонн", "▁ачаа", "▁тээвэрлэсэн", "."],
                "Morphological": ["Мөн", "өнгөрсөн", "жил", "сая", "тонн", "ачаа", "тээвэрлэсэн", "."]
            }
        }
    ]

    for item in data:
        print(f"\n{'='*40}")
        print(f"Line {item['line']}")
        print(f"{'-'*40}")
        print(f"Original Text: {item['original']}\n")
        
        for model, tokens in item["tokenizations"].items():
            print(f"{model}:")
            print("  " + " | ".join(tokens))
            print("-"*40)
            
        print("="*40)

if __name__ == "__main__":
    print_tokenizations()
